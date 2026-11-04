/*
  SMART INSOLE - COMPLETE EDGE-AI CODE
  1. Trained AI Weights & Ulcer Dataset Stored in ESP32
  2. Reads Live Sensor Values (5 FSR + BMP280 + MPU6500)
  3. Compares Stored AI/Dataset with Live Sensors
  4. Shows Neat Result on OLED + Serial Monitor
*/
#include <Wire.h>
#include <Adafruit_BMP280.h>
#include <U8g2lib.h>
#include <math.h>
// ============================================================================
// PART 1: TRAINED AI WEIGHTS & ULCER DATASET STORED IN ESP32 MEMORY
// (Trained in Python from 1,500-row dfu_patient_dataset.csv)
// ============================================================================
const float NN_W1[8][7] = {
  {  0.034f, -0.261f, -0.240f, -0.265f, -0.067f, -0.235f, -0.251f },
  { -0.054f, -0.329f, -0.196f,  0.003f, -0.331f, -0.210f,  0.031f },
  {  0.098f, -1.093f, -1.228f, -1.004f,  0.002f, -1.334f, -0.648f },
  {  0.346f,  1.296f,  1.394f,  1.295f,  0.565f,  1.420f,  0.895f },
  {  0.143f, -0.317f, -0.190f, -0.147f, -0.294f, -0.187f, -0.155f },
  {  0.447f,  1.774f,  1.935f,  1.796f,  0.798f,  1.998f,  1.278f },
  {  0.236f,  0.190f, -0.191f, -0.328f, -0.130f, -0.162f,  0.308f },
  {  0.068f,  0.184f,  0.265f,  0.285f,  0.096f,  0.225f,  0.162f }
};
const float NN_B1[8] = { 0.600f, 0.000f, 6.398f, -2.030f, 0.000f, -2.785f, -0.005f, -0.392f };

const float NN_W2[3][8] = {
  {  0.364f, 0.002f,  1.963f, -2.038f,  0.020f, -2.781f, -0.021f, -0.375f },
  { -0.314f, 0.042f,  0.697f,  0.114f, -0.030f,  0.245f, -0.019f,  0.017f },
  {  0.016f, 0.003f, -2.646f,  1.831f, -0.016f,  2.614f,  0.035f,  0.314f }
};
const float NN_B2[3] = { 1.713f, 1.214f, -2.927f };

// Reference Ulcer Patient Dataset stored in ESP32 Flash (PROGMEM)
struct PatientRecord {
  uint16_t fsr[5];
  float temp_delta;
  float tilt_deg;
  uint8_t risk_lvl; // 0 = LOW, 1 = MEDIUM, 2 = HIGH (Ulcer)
};

const int DATASET_SIZE = 9;
const PatientRecord ULCER_PATIENT_DB[DATASET_SIZE] PROGMEM = {
  // Normal / Safe Standing Profiles (Risk = 0)
  { {   50,   40,   45,   40,   35 }, 0.1f,  1.0f, 0 },
  { { 1650, 1350, 1480, 1390, 1150 }, 0.3f,  2.5f, 0 },
  { { 3100, 2850, 3050, 2900, 2600 }, 0.6f,  4.0f, 0 },
  // Medium Pre-Ulcer Profiles (Risk = 1)
  { {  900, 2750, 1100,  900,  600 }, 1.4f,  9.0f, 1 },
  { { 1800, 2100, 2200, 1900, 1500 }, 1.6f,  6.0f, 1 },
  { { 1700, 1900, 2100, 2000, 1800 }, 0.8f, 14.5f, 1 },
  // High-Risk Ulcer Patient Profiles (Risk = 2)
  { {  250, 3750,  450,  300,  200 }, 1.2f,  6.0f, 2 },
  { {  300,  500, 3800,  400,  250 }, 2.4f,  8.0f, 2 },
  { { 1900, 3100, 3400, 3100, 2200 }, 2.3f, 21.0f, 2 }
};

// ---------- HARDWARE OBJECTS & PINS ----------
U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);
Adafruit_BMP280 bmp;
bool bmpOK = false;

#define MPU_ADDR 0x68
bool mpuOK = false;

const int fsrPins[5] = {34, 32, 35, 33, 39};
const char* zoneNames[5] = {"Heel", "Big Toe", "Meta-1", "Meta-3", "Meta-5"};
int fsrValues[5] = {0, 0, 0, 0, 0};

float ax = 0, ay = 0, az = 0;
float gx = 0, gy = 0, gz = 0;
float temperatureC = 32.0f;

float baselineTempC = 32.0f;
float baselineRollDeg = 0.0f;
float tempDeltaC = 0.0f;
float tiltDev = 0.0f;

uint8_t currentRiskLevel = 0; // 0 = LOW, 1 = MEDIUM, 2 = HIGH (ULCER ALERT)
int peakFSR = 0;
int peakZoneIdx = 0;
int highRiskStreak = 0;
const int ALERT_STREAK_REQUIRED = 8;
bool alertBlinkState = false;

unsigned long lastSample = 0;
const unsigned long SAMPLE_INTERVAL = 200; // 5 Hz

// =====================================================
// MPU6500 REGISTER FUNCTIONS
// =====================================================

void mpuWriteRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

bool mpuReadRegisters(uint8_t reg, uint8_t count, uint8_t *buffer) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;

  uint8_t received = Wire.requestFrom((uint8_t)MPU_ADDR, count, (uint8_t)true);
  if (received != count) {
    while (Wire.available()) Wire.read();
    return false;
  }
  for (uint8_t i = 0; i < count; i++) buffer[i] = Wire.read();
  return true;
}

int16_t combineBytes(uint8_t highByte, uint8_t lowByte) {
  return (int16_t)(((uint16_t)highByte << 8) | lowByte);
}

bool setupMPU6500() {
  uint8_t whoAmI = 0;
  if (!mpuReadRegisters(0x75, 1, &whoAmI)) return false;
  if (whoAmI != 0x70 && whoAmI != 0x68 && whoAmI != 0x71) return false;

  mpuWriteRegister(0x6B, 0x00);
  delay(100);
  mpuWriteRegister(0x1C, 0x00);
  mpuWriteRegister(0x1B, 0x00);
  delay(50);
  return true;
}

void readMPU6500() {
  uint8_t data[14];
  if (!mpuReadRegisters(0x3B, 14, data)) {
    mpuOK = false;
    return;
  }
  ax = combineBytes(data[0], data[1]) / 16384.0f;
  ay = combineBytes(data[2], data[3]) / 16384.0f;
  az = combineBytes(data[4], data[5]) / 16384.0f;
  gx = combineBytes(data[8], data[9]) / 131.0f;
  gy = combineBytes(data[10], data[11]) / 131.0f;
  gz = combineBytes(data[12], data[13]) / 131.0f;
  mpuOK = true;
}
// =====================================================
// CALIBRATE BASELINE AT STARTUP
// =====================================================
void calibrateBaseline() {
  u8g2.clearBuffer();
  u8g2.drawFrame(0, 0, 128, 64);
  u8g2.setFont(u8g2_font_7x13B_tf);
  u8g2.drawStr(16, 26, "SMART SOLE AI");
  u8g2.setFont(u8g2_font_6x12_tf);
  u8g2.drawStr(22, 46, "Calibrating...");
  u8g2.sendBuffer();
  Serial.println("# Calibrating user baseline...");
  float sumTemp = 0;
  float sumRoll = 0;
  const int N = 12;
  for (int s = 0; s < N; s++) {
    readMPU6500();
    sumRoll += atan2f(ay, az) * 57.2958f;
    sumTemp += bmpOK ? bmp.readTemperature() : 32.0f;
    delay(100);
  }
  baselineTempC = sumTemp / N;
  baselineRollDeg = sumRoll / N;
}
// =====================================================
// PART 2: COMPARE STORED AI & DATASET WITH LIVE SENSORS
// =====================================================
// A. Run Stored TinyML Neural Network Weights on Live Sensor Data
uint8_t runTinyMLModel() {
  float x[7] = {
    fsrValues[0] / 4095.0f,
    fsrValues[1] / 4095.0f,
    fsrValues[2] / 4095.0f,
    fsrValues[3] / 4095.0f,
    fsrValues[4] / 4095.0f,
    tempDeltaC / 4.0f,
    tiltDev / 35.0f
  };

  float hidden[8];
  for (int i = 0; i < 8; i++) {
    float sum = NN_B1[i];
    for (int j = 0; j < 7; j++) sum += NN_W1[i][j] * x[j];
    hidden[i] = (sum > 0.0f) ? sum : 0.0f;
  }

  int bestClass = 0;
  float bestScore = -999.0f;
  for (int k = 0; k < 3; k++) {
    float sum = NN_B2[k];
    for (int i = 0; i < 8; i++) sum += NN_W2[k][i] * hidden[i];
    if (sum > bestScore) {
      bestScore = sum;
      bestClass = k;
    }
  }
  return bestClass;
}
// B. Compare Live Sensor Data with Stored Ulcer Patient Dataset
uint8_t compareWithStoredDataset() {
  float minDist = 9999.0f;
  uint8_t matchedRisk = 0;

  for (int i = 0; i < DATASET_SIZE; i++) {
    PatientRecord ref;
    memcpy_P(&ref, &ULCER_PATIENT_DB[i], sizeof(PatientRecord));

    float d_fsr = 0.0f;
    for (int j = 0; j < 5; j++) {
      float diff = (float(fsrValues[j]) - float(ref.fsr[j])) / 4095.0f;
      d_fsr += diff * diff;
    }
    float d_temp = (tempDeltaC - ref.temp_delta) / 4.0f;
    float d_tilt = (tiltDev - ref.tilt_deg) / 30.0f;

    float dist = sqrtf(d_fsr + 2.0f * d_temp * d_temp + 1.5f * d_tilt * d_tilt);
    if (dist < minDist) {
      minDist = dist;
      matchedRisk = ref.risk_lvl;
    }
  }
  return matchedRisk;
}
// C. Final Edge-AI Fusion Decision
void evaluateUlcerRisk() {
  tempDeltaC = bmpOK ? (temperatureC - baselineTempC) : 0.0f;
  float rollDeg = atan2f(ay, az) * 57.2958f;
  tiltDev = fabsf(rollDeg - baselineRollDeg);
  peakFSR = 0;
  peakZoneIdx = 0;
  long sumFSR = 0;
  int activeCount = 0;
  for (int i = 0; i < 5; i++) {
    sumFSR += fsrValues[i];
    if (fsrValues[i] > 1200) activeCount++;
    if (fsrValues[i] > peakFSR) {
      peakFSR = fsrValues[i];
      peakZoneIdx = i;
    }
  }
  float avgOtherFSR = (sumFSR - peakFSR) / 4.0f;
  // Get predictions from Stored TinyML Model & Stored Patient Dataset
  uint8_t nnRisk      = runTinyMLModel();
  uint8_t datasetRisk = compareWithStoredDataset();
  // Spatial Load Balance Checks (Prevents false alarm on normal balanced standing)
  bool focalHotspot     = (peakFSR >= 3300) && (avgOtherFSR < 900.0f) && (activeCount <= 2);
  bool pressAndTemp     = (peakFSR >= 2600) && (tempDeltaC >= 2.0f);
  bool pressAndTilt     = (peakFSR >= 2800) && (tiltDev >= 18.0f);
  bool moderateOverload = (peakFSR >= 2700) && (avgOtherFSR < 1300.0f);
  bool moderateTemp     = (tempDeltaC >= 1.2f);

  uint8_t instantRisk = 0;
  if (focalHotspot || pressAndTemp || pressAndTilt ||
      ((nnRisk == 2 || datasetRisk == 2) && (activeCount <= 2 || tempDeltaC >= 1.8f))) {
    instantRisk = 2;
  } else if (datasetRisk == 1 || moderateOverload || moderateTemp || tiltDev >= 12.0f) {
    instantRisk = 1;
  } else {
    instantRisk = 0;
  }

  if (instantRisk == 2) {
    highRiskStreak++;
    currentRiskLevel = (highRiskStreak >= ALERT_STREAK_REQUIRED) ? 2 : 1;
  } else {
    if (highRiskStreak > 0) highRiskStreak--;
    currentRiskLevel = instantRisk;
  }
}
// =====================================================
// READ SENSORS
// =====================================================
void readSensors() {
  for (int i = 0; i < 5; i++) {
    fsrValues[i] = analogRead(fsrPins[i]);
  }
  readMPU6500();
  if (bmpOK) {
    temperatureC = bmp.readTemperature();
  }
  evaluateUlcerRisk();
}
// =====================================================
// SERIAL MONITOR OUTPUT
// =====================================================
void printSerialHeader() {
  Serial.println("time_ms,FSR1_Heel,FSR2_BigToe,FSR3_M1,FSR4_M3,FSR5_M5,Peak_Spot,Temp_C,TempDelta_C,Tilt_Deg,Risk_Level,Status");
}

void printSerialData() {
  const char* statusText[3] = {"LOW_SAFE", "MEDIUM_WARN", "!!_ULCER_ALERT_!!"};

  Serial.print(millis());
  Serial.print(",");
  Serial.print(fsrValues[0]);
  Serial.print(",");
  Serial.print(fsrValues[1]);
  Serial.print(",");
  Serial.print(fsrValues[2]);
  Serial.print(",");
  Serial.print(fsrValues[3]);
  Serial.print(",");
  Serial.print(fsrValues[4]);
  Serial.print(",");
  Serial.print((peakFSR > 300) ? zoneNames[peakZoneIdx] : "No_Load");
  Serial.print(",");
  Serial.print(temperatureC, 2);
  Serial.print(",");
  Serial.print(tempDeltaC, 2);
  Serial.print(",");
  Serial.print(tiltDev, 1);
  Serial.print(",");
  Serial.print(currentRiskLevel);
  Serial.print(",");
  Serial.println(statusText[currentRiskLevel]);
}
// =====================================================
// SIMPLE, NEAT SINGLE-SCREEN OLED DISPLAY (WITH TILT)
// =====================================================
void displayOLED() {
  char line[32];
  u8g2.clearBuffer();

  // ---------------- ULCER ALERT SCREEN ----------------
  if (currentRiskLevel == 2) {
    alertBlinkState = !alertBlinkState;

    if (alertBlinkState) {
      u8g2.drawBox(0, 0, 128, 16);
      u8g2.setDrawColor(0);
      u8g2.setFont(u8g2_font_7x13B_tf);
      u8g2.drawStr(8, 13, "!! ULCER ALERT !!");
      u8g2.setDrawColor(1);
    } else {
      u8g2.drawFrame(0, 0, 128, 16);
      u8g2.setFont(u8g2_font_7x13B_tf);
      u8g2.drawStr(8, 13, "!! ULCER ALERT !!");
    }

    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(4, 28, "RISK : HIGH RISK");

    snprintf(line, sizeof(line), "SPOT : %s (%d)", zoneNames[peakZoneIdx], peakFSR);
    u8g2.drawStr(4, 39, line);

    snprintf(line, sizeof(line), "TEMP : %.1f C", temperatureC);
    u8g2.drawStr(4, 50, line);

    snprintf(line, sizeof(line), "TILT : %.0f deg (%s)",
             tiltDev, (tiltDev >= 15.0f) ? "Tilted" : "Normal");
    u8g2.drawStr(4, 61, line);

    u8g2.sendBuffer();
    return;
  }

  // ---------------- NORMAL / MEDIUM SCREEN ----------------
  u8g2.drawFrame(0, 0, 128, 64);
  u8g2.drawHLine(0, 15, 128);

  u8g2.setFont(u8g2_font_7x13B_tf);
  u8g2.drawStr(16, 12, "SMART SOLE AI");

  u8g2.setFont(u8g2_font_6x10_tf);

  if (currentRiskLevel == 1) {
    u8g2.drawStr(5, 27, "RISK : MEDIUM");
  } else {
    u8g2.drawStr(5, 27, "RISK : LOW (SAFE)");
  }

  if (peakFSR > 300) {
    snprintf(line, sizeof(line), "PRESS: %d (%s)", peakFSR, zoneNames[peakZoneIdx]);
  } else {
    snprintf(line, sizeof(line), "PRESS: 0 (No Load)");
  }
  u8g2.drawStr(5, 38, line);

  snprintf(line, sizeof(line), "TEMP : %.1f C", temperatureC);
  u8g2.drawStr(5, 49, line);

  snprintf(line, sizeof(line), "TILT : %.0f deg (%s)",
           tiltDev, (tiltDev >= 12.0f) ? "Tilted" : "Normal");
  u8g2.drawStr(5, 60, line);

  u8g2.sendBuffer();
}
// =====================================================
// SETUP
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(300);

  Wire.begin(21, 22);
  Wire.setClock(100000);

  analogReadResolution(12);
  for (int i = 0; i < 5; i++) {
    analogSetPinAttenuation(fsrPins[i], ADC_11db);
  }

  u8g2.begin();

  bmpOK = bmp.begin(0x76);
  if (!bmpOK) bmpOK = bmp.begin(0x77);

  mpuOK = setupMPU6500();

  calibrateBaseline();
  printSerialHeader();
}
void loop() {
  unsigned long now = millis();

  if (now - lastSample >= SAMPLE_INTERVAL) {
    lastSample = now;
    readSensors();
    printSerialData();
    displayOLED();
  }
}