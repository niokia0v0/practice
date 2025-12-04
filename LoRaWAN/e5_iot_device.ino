#include <Arduino.h>
#include <HardwareSerial.h>
#include "DHT.h"

// --- Hardware Pin Definitions ---
#define DHTPIN 2       // DHT11 data pin is connected to GPIO2
#define DHTTYPE DHT11  // Sensor type is DHT11

// --- LoRa-E5 UART Definition ---
// Using HardwareSerial(1) on ESP32-S3
// Default UART1 pins for ESP32-S3 are: RX = GPIO9, TX = GPIO10
// Need to connect the LoRa module's TX pin to GPIO9 and its RX pin to GPIO10.
HardwareSerial loraSerial(1);

// --- Sensor Object Initialization ---
DHT dht(DHTPIN, DHTTYPE);

// --- TTN Device Credentials (Replace with actual values from TTN) ---
const char* devEui = "**********"; // Get this from LoRa module's label
const char* appEui = "**********"; // This is the JoinEUI. Replace with the one set in TTN.
const char* appKey = "**********"; // 16-byte (32-char) AppKey, padded with zeros if necessary.

// Function: Sends an AT command to the LoRa-E5 module and prints the response.
void sendATCommand(const char* cmd, int wait = 1000) {
    Serial.print(">> "); // Print command to the serial monitor for debugging
    Serial.println(cmd);
    
    loraSerial.println(cmd); // Send command to the LoRa module
    
    long startTime = millis();
    while (millis() - startTime < wait) { // Wait for a response
        if (loraSerial.available()) {
            Serial.write(loraSerial.read());
        }
    }
}

void setup() {
    Serial.begin(115200); // Initialize serial communication for debugging output
    delay(1000);
    Serial.println("--- LoRaWAN Node Start ---");

    // Initialize serial communication with the LoRa-E5 module
    // RX pin, TX pin for UART1 on ESP32-S3
    loraSerial.begin(9600, SERIAL_8N1, 9, 10); 

    // Initialize the DHT11 sensor
    dht.begin();
    delay(2000);

    // --- Configure the LoRa-E5 module (this configuration only needs to run once successfully) ---
    Serial.println("Configuring LoRa-E5 for TTN (OTAA)...");

    sendATCommand("AT"); // Test communication with the module
    delay(100);

    // Set the device credentials
    char cmd_buffer[128];
    sprintf(cmd_buffer, "AT+ID=DevEui,\"%s\"", devEui);
    sendATCommand(cmd_buffer);

    sprintf(cmd_buffer, "AT+ID=AppEui,\"%s\"", appEui);
    sendATCommand(cmd_buffer);

    sprintf(cmd_buffer, "AT+KEY=APPKEY,\"%s\"", appKey);
    sendATCommand(cmd_buffer);

    // Set the mode to LoRaWAN OTAA (Over-The-Air Activation)
    sendATCommand("AT+MODE=LWOTAA");

    // Set the frequency plan to Europe 868MHz
    sendATCommand("AT+DR=EU868");
    
    // Set the Data Rate. DR5 (SF7) is a good starting point for fast transmission.
    // You can change this to DR0 (SF12) for longer range.
    sendATCommand("AT+DR=5");

    // Set the channels for EU868 (default is channels 0, 1, 2)
    sendATCommand("AT+CH=NUM,0-2");

    // Set the device class to Class A (most power-efficient)
    sendATCommand("AT+CLASS=A");
    
    // Set the port for data messages (e.g., port 1)
    sendATCommand("AT+PORT=1");

    // Save the configuration to the module's non-volatile memory
    sendATCommand("AT+SAVE");
    delay(200); // Add a small delay after saving
    
    // Attempt to join the network
    Serial.println("Attempting to join the network...");
    sendATCommand("AT+JOIN", 8000); // Use a longer timeout (8 seconds) to wait for the join process
}

void loop() {
    // Read sensor data
    float temperature = dht.readTemperature();
    float humidity = dht.readHumidity();

    // Check if sensor reading was successful
    if (isnan(temperature) || isnan(humidity)) {
        Serial.println("Failed to read from DHT sensor!");
    } else {
        Serial.print("Temperature: ");
        Serial.print(temperature);
        Serial.print(" °C, Humidity: ");
        Serial.print(humidity);
        Serial.println(" %");

        // Prepare the payload to be sent
        // We encode the data into bytes for efficiency. 
        // Example: temperature * 10, humidity * 10
        byte payload[4];
        
        // Scale and convert temperature to a 16-bit signed integer
        int16_t temp_scaled = (int16_t)(temperature * 10);
        // Scale and convert humidity to a 16-bit unsigned integer
        uint16_t hum_scaled = (uint16_t)(humidity * 10);

        // Split the 16-bit values into two 8-bit bytes (Big Endian)
        payload[0] = highByte(temp_scaled);
        payload[1] = lowByte(temp_scaled);
        payload[2] = highByte(hum_scaled);
        payload[3] = lowByte(hum_scaled);

        // Construct the AT command to send the payload in hexadecimal format
        char cmd_buffer[50];
        sprintf(cmd_buffer, "AT+MSGHEX=\"%02X%02X%02X%02X\"", payload[0], payload[1], payload[2], payload[3]);
        
        Serial.println("Sending data...");
        sendATCommand(cmd_buffer, 10000); // Send data and wait up to 10 seconds for a response
    }

    // Wait for 1 minute(60000 ms) before the next transmission
    Serial.println("Waiting for 1 minute...");
    delay(60000); 
}