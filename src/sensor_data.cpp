#include <LSM303AGR_ACC_Sensor.h>
#include <Wire.h>

LSM303AGR_ACC_Sensor Acc(&Wire);

void setup() {
  Serial.begin(115200);
  Wire.begin(8, 9);   // IndusBoard / Coin V2 pins

  if (Acc.begin() != 0) {
    Serial.println("Accelerometer not found");
    while (1);
  }

  Acc.Enable();

  // IMPORTANT:
  // This library does NOT allow changing ODR.
  // We will assume default ~100 Hz and control sampling in software.

  Serial.println("Acc_X,Acc_Y,Acc_Z");
}

void loop() {
  int32_t acc[3];
  Acc.GetAxes(acc);

  Serial.print(acc[0]); Serial.print(",");
  Serial.print(acc[1]); Serial.print(",");
  Serial.println(acc[2]);

  // ~100 Hz sampling (10 ms)
  delay(10);
}
