#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/i2c.h"
#include "esp_adc/adc_oneshot.h"

// Bibliotecas do TensorFlow Lite Micro
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h" // Substituiu o all_ops_resolver
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define ASL_MODEL_IMPLEMENTATION
// Inclusão dos cabeçalhos gerados pelo seu script_luva.ipynb
#include "modelo_gru_quantizado.h" // Seu modelo C-Array
#include "scaler_params.h"         // Parâmetros do MinMaxScaler e StandardScaler

static const char *TAG = "ASL_GLOVE_INFERENCIA";

// ==========================================
// CONFIGURAÇÕES DE HARDWARE (ESP32-S3)
// ==========================================
#define I2C_MASTER_SCL_IO           2
#define I2C_MASTER_SDA_IO           1
#define I2C_MASTER_NUM              I2C_NUM_0
#define I2C_MASTER_FREQ_HZ          400000
#define MPU6050_ADDR                0x68

const adc_channel_t flex_channels[5] = {
    ADC_CHANNEL_3, ADC_CHANNEL_4, ADC_CHANNEL_5, ADC_CHANNEL_6, ADC_CHANNEL_7
};
adc_oneshot_unit_handle_t adc1_handle;

// ==========================================
// CONFIGURAÇÕES DO MODELO
// ==========================================
#define NUM_FEATURES 11    
#define SEQUENCE_LENGTH 50 
#define NUM_CLASSES 40     

#define MPU_ACCEL_SCALE  (9.81f / 16384.0f)          
#define MPU_GYRO_SCALE   ((float)M_PI / (131.0f * 180.0f)) 

float sequence_buffer[SEQUENCE_LENGTH][NUM_FEATURES];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

constexpr int kTensorArenaSize = 160 * 1024; 
uint8_t tensor_arena[kTensorArenaSize];

// ==========================================
// FUNÇÕES DE HARDWARE
// ==========================================
void init_hardware() {
    adc_oneshot_unit_init_cfg_t init_config = { .unit_id = ADC_UNIT_1 };
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&init_config, &adc1_handle));

    adc_oneshot_chan_cfg_t config = {
        .atten = ADC_ATTEN_DB_11, 
        .bitwidth = ADC_BITWIDTH_12,
    };
    for (int i = 0; i < 5; i++) {
        ESP_ERROR_CHECK(adc_oneshot_config_channel(adc1_handle, flex_channels[i], &config));
    }

    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master = { .clk_speed = I2C_MASTER_FREQ_HZ },
    };
    i2c_param_config(I2C_MASTER_NUM, &conf);
    i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);

    uint8_t cmd[2] = {0x6B, 0x00};
    i2c_master_write_to_device(I2C_MASTER_NUM, MPU6050_ADDR, cmd, 2, 1000 / portTICK_PERIOD_MS);
    
    ESP_LOGI(TAG, "Sensores Inicializados.");
}

void read_and_normalize_sensors(float* current_frame) {
    // 1. LEITURA E NORMALIZAÇÃO DOS FLEXORES (Com aplicação estrita do Clamp)
    for (int i = 0; i < 5; i++) {
        int raw_val;
        adc_oneshot_read(adc1_handle, flex_channels[i], &raw_val);
        
        // Aplica a fórmula do MinMaxScaler contida no scaler_params.h
        float norm = ((float)raw_val - ASL_FLEX_MIN[i]) / ASL_FLEX_RANGE[i];
        
        // Garante que o valor fique estritamente entre 0.0f e 1.0f
        if (norm < 0.0f) norm = 0.0f;
        if (norm > 1.0f) norm = 1.0f;
        
        current_frame[i] = norm;
    }

// 2. LEITURA DOS DADOS BRUTOS DA IMU VIA I2C
    uint8_t reg = 0x3B; 
    uint8_t data[14];
    i2c_master_write_read_device(I2C_MASTER_NUM, MPU6050_ADDR, &reg, 1, data, 14, 1000 / portTICK_PERIOD_MS);

    // Valores crus (RAW) exatamente como saem da sua placa
    int16_t acc_x_raw = (data[0] << 8) | data[1];
    int16_t acc_y_raw = (data[2] << 8) | data[3];
    int16_t acc_z_raw = (data[4] << 8) | data[5];
    int16_t gyr_x_raw = (data[8] << 8) | data[9];
    int16_t gyr_y_raw = (data[10] << 8) | data[11];
    int16_t gyr_z_raw = (data[12] << 8) | data[13];

    // ==========================================================
    // CORREÇÃO DE EIXOS + CONVERSÃO PARA m/s² e rad/s
    // ==========================================================
    // O modelo espera a gravidade no Eixo X, mas sua luva tem no Eixo Z.
    // Trocamos X pelo Z. (Mantemos o Y no lugar dele).
    float imu_physical[6] = {
        (float)acc_z_raw * MPU_ACCEL_SCALE, // Vai para ACCx do Modelo
        (float)acc_y_raw * MPU_ACCEL_SCALE, // Vai para ACCy do Modelo
        (float)acc_x_raw * MPU_ACCEL_SCALE, // Vai para ACCz do Modelo
        
        // O Giroscópio precisa acompanhar a mesma troca de eixos!
        (float)gyr_z_raw * MPU_GYRO_SCALE,  // Vai para GYRx do Modelo
        (float)gyr_y_raw * MPU_GYRO_SCALE,  // Vai para GYRy do Modelo
        (float)gyr_x_raw * MPU_GYRO_SCALE   // Vai para GYRz do Modelo
    };

    // 3. NORMALIZAÇÃO DA IMU (StandardScaler)
    for (int i = 0; i < 6; i++) {
        current_frame[5 + i] = (imu_physical[i] - ASL_IMU_MEAN[i]) / ASL_IMU_STD[i];
    }
}

// ==========================================
// TAREFA PRINCIPAL DO ESP32
// ==========================================
extern "C" void app_main() {
    init_hardware();

    model = tflite::GetModel(modelo_gru_quantizado_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Versão do TFLite incompatível!");
        return;
    }

    // --- O GRANDE SEGREDO: REGISTRAR AS OPERAÇÕES MANUAIS ---
    // Uma rede GRU unrolled é desmontada nestas operações matemáticas básicas
    static tflite::MicroMutableOpResolver<25> resolver;
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddSplitV();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddSub();
    resolver.AddLogistic(); // Equivale ao Sigmoid
    resolver.AddTanh();
    resolver.AddConcatenation();
    resolver.AddSplit();
    resolver.AddPack();
    resolver.AddUnpack();
    resolver.AddStridedSlice();
    resolver.AddGather();

    // Instancia o Interpretador usando o nosso resolver manual
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Faltou memória RAM (Tensor Arena)! Aumente kTensorArenaSize.");
        return;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    float input_scale = input_tensor->params.scale;
    int input_zero_point = input_tensor->params.zero_point;
    float output_scale = output_tensor->params.scale;
    int output_zero_point = output_tensor->params.zero_point;

    ESP_LOGI(TAG, "Modelo carregado com sucesso.");

    float current_frame[NUM_FEATURES];

    // ==========================================================
    // 1. ENCHENDO O BUFFER INICIAL (Evita lixo na inicialização)
    // ==========================================================
    ESP_LOGI(TAG, "Calibrando janela de leitura (Aguarde 2.5s)...");
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
        read_and_normalize_sensors(current_frame);
        memcpy(sequence_buffer[i], current_frame, sizeof(float) * NUM_FEATURES);
        vTaskDelay(pdMS_TO_TICKS(50));
    }
    ESP_LOGI(TAG, "✅ PRONTO! Pode começar a fazer os gestos.");


    int cooldown_timer = 0; // Variável que controla o "silêncio" da luva

    while (true) {
        // Atualiza a janela deslizante de sensores O TEMPO TODO
        read_and_normalize_sensors(current_frame);
        for (int i = 0; i < SEQUENCE_LENGTH - 1; i++) {
            memcpy(sequence_buffer[i], sequence_buffer[i + 1], sizeof(float) * NUM_FEATURES);
        }
        memcpy(sequence_buffer[SEQUENCE_LENGTH - 1], current_frame, sizeof(float) * NUM_FEATURES);


        // ==========================================================
        // 2. LÓGICA DE COOLDOWN E GATILHO DE GESTOS
        // ==========================================================
        if (cooldown_timer > 0) {
            cooldown_timer--; // Mão em transição/repouso, não faz inferência!
        } 
        else {
            // Copia dados para o tensor e roda a Rede Neural
            int8_t* input_data = input_tensor->data.int8;
            int tensor_index = 0;
            
            for (int i = 0; i < SEQUENCE_LENGTH; i++) {
                for (int j = 0; j < NUM_FEATURES; j++) {
                    float val = sequence_buffer[i][j];
                    int8_t quantized_val = (int8_t)(round(val / input_scale) + input_zero_point);
                    input_data[tensor_index++] = quantized_val;
                }
            }

            if (interpreter->Invoke() != kTfLiteOk) {
                ESP_LOGE(TAG, "Falha durante o Invoke()!");
                continue;
            }

            // Procura a classe vencedora
            int8_t* output_data = output_tensor->data.int8;
            float max_prob = 0.0f;
            int winning_class = -1;

            for (int i = 0; i < NUM_CLASSES; i++) {
                float prob = (output_data[i] - output_zero_point) * output_scale;
                if (prob > max_prob) {
                    max_prob = prob;
                    winning_class = i;
                }
            }

            // Se o modelo tem muita certeza do gesto (> 85%), ele imprime!
            if (max_prob > 0.85f) { 
                ESP_LOGI(TAG, "========================================");
                ESP_LOGI(TAG, "🌟 GESTO TRADUZIDO: Classe %d", winning_class);
                ESP_LOGI(TAG, "   Confiança: %.1f%%", max_prob * 100);
                ESP_LOGI(TAG, "========================================");
                
                // Ativa o Cooldown: Trava a rede neural por 1.5 segundos (30 loops de 50ms)
                // Isso dá tempo para você desfazer o gesto sem gerar traduções falsas.
                cooldown_timer = 30; 
            }
        }

        vTaskDelay(pdMS_TO_TICKS(50)); 
    }
}