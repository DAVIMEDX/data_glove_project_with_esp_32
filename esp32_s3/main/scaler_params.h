// ==========================================================================
// scaler_params.h  —  Parâmetros de normalização híbrida para ASL Glove
// Gerado automaticamente pelo pipeline Python (Gargalo 1)
// NÃO EDITAR MANUALMENTE
// ==========================================================================
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#define ASL_N_FEATURES  11
#define ASL_WINDOW_SIZE 50
#define ASL_N_CLASSES   40

// Fórmula: norm = clamp((raw - FLEX_MIN[i]) / FLEX_RANGE[i], 0.0f, 1.0f)
// ── MinMaxScaler para sensores de FLEXÃO (índices 0..4) ──────────────────────
// Fórmula: norm = clamp((raw - FLEX_MIN[i]) / FLEX_RANGE[i], 0.0f, 1.0f)

static const float ASL_FLEX_MIN[5]   = { 2450.0f, 2450.0f, 2450.0f, 2450.0f, 2450.0f };
static const float ASL_FLEX_RANGE[5] = { 650.0f, 650.0f, 650.0f, 650.0f, 650.0f };

// Fórmula: norm = (raw - IMU_MEAN[i]) / IMU_STD[i]
static const float ASL_IMU_MEAN[6] = { 6.30664509f,  2.57584966f,  1.66420334f,
                                        0.01113935f,  0.00814702f, -0.01568443f };
static const float ASL_IMU_STD[6]  = { 4.43173557f,  3.93960572f,  4.36593563f,
                                        0.31848653f,  0.45113991f,  0.26912546f };

#ifdef __cplusplus
}
#endif