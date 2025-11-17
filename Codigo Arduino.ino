#include <WiFi.h>
#include <WiFiUdp.h>

// Bibliotecas do IMU (ex: MPU6050)
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

// --- [CONFIGURAÇÕES DE REDE - EDITE AQUI!] ---
const char* ssid = "Cecília"; // Coloque o nome do seu Wi-Fi
const char* password = "cececici"; // Coloque sua senha

// IP do computador que vai rodar o Python (que você anotou acima)
const char* host_ip = "172.20.10.7";
// Porta de comunicação (tem que ser a mesma no Python)
const int udp_port = 4210; 
// ---------------------------------------------

// --- [CONFIGURAÇÕES DE HARDWARE] ---
Adafruit_MPU6050 mpu;
const int pinoSEMG = 4; // Pino do sEMG

// [NOVO] Defina os pinos I2C que vamos usar
#define SDA_PIN 10
#define SCL_PIN 11
// ---------------------------------

// [NOVO] Objeto para a comunicação UDP
WiFiUDP udp;

void setup() {
  Serial.begin(115200);
  delay(2000); 
  Wire.begin(SDA_PIN, SCL_PIN);

  // --- Inicialização do IMU ---
  if (!mpu.begin(0x68)) {
    Serial.println("Falha ao encontrar o sensor MPU6050.");
    while (1) { delay(10); }
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  
  // --- [NOVO] Conexão Wi-Fi ---
  Serial.println();
  Serial.print("Conectando a ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  int tentativas = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    tentativas++;
    if (tentativas > 20) {
      Serial.println("\nFalha ao conectar ao Wi-Fi. Reiniciando...");
      ESP.restart();
    }
  }

  Serial.println("\nWi-Fi conectado!");
  Serial.print("Endereço IP do ESP32: ");
  Serial.println(WiFi.localIP());
  Serial.print("Enviando dados para ");
  Serial.print(host_ip);
  Serial.print(":");
  Serial.println(udp_port);
  Serial.println("Dispositivo pronto para enviar dados (IMU + sEMG).");
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  int valorSEMG = analogRead(pinoSEMG);

  // [NOVO] Formata os dados em uma única string
  // Usamos um buffer de char (mais rápido que String no loop)
  char buffer_dados[100]; 
  
  // Constrói a string no formato "Ax,Ay,Az,Gx,Gy,Gz,sEMG"
  sprintf(buffer_dados, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d",
    a.acceleration.x,
    a.acceleration.y,
    a.acceleration.z,
    g.gyro.x,
    g.gyro.y,
    g.gyro.z,
    valorSEMG
  );

  // [NOVO] Envio dos dados via UDP
  udp.beginPacket(host_ip, udp_port); // Prepara o pacote
  udp.print(buffer_dados);            // Escreve os dados no pacote
  udp.endPacket();                    // Envia o pacote pela rede

  // O delay controla a taxa de amostragem
  delay(1); 
}
