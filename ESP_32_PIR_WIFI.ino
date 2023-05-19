#include <iostream>
#include <stdlib.h>
#include <string>
#include <AsyncTCP.h>
#ifdef ESP32
#include <WiFi.h>
#include <AsyncTCP.h>
#elif defined(ESP8266)
#include <ESP8266WiFi.h>
#include <ESPAsyncTCP.h>
#endif
#include <ESPAsyncWebServer.h>
#include "ArduinoJson.h"
#include <TinyGPSPlus.h>
#include <SoftwareSerial.h>

const int wifi_led = 2;//Pin of LED

const int pinPIR = 13; //Pin of PIR

const char * ssid = "CLOUDCAM_e051d8fd4a58";//WiFi name
const char * password = "";//WiFi password

const char * PARAM_DIOD = "signal";

unsigned long previousMillis = 0;
unsigned long interval = 30000;

void notFound(AsyncWebServerRequest *request) {
    request->send(404, "text/plain", "Not found");
}

AsyncWebServer server(8080);

void setup()
{
    Serial.begin(115200);
    pinMode(wifi_led, OUTPUT);
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    Serial.print("Connecting");
    while (WiFi.waitForConnectResult() != WL_CONNECTED) {
      digitalWrite(wifi_led, HIGH);
      delay(500);
      digitalWrite(wifi_led, LOW);
      delay(500);
    }
    Serial.println();
    if (WiFi.waitForConnectResult() == WL_CONNECTED)
    {
      Serial.println("Connected");
      Serial.println(WiFi.localIP());
    }
    server.on("/diod", HTTP_GET, [] (AsyncWebServerRequest *request) {
        String speed;
        if (request->hasParam( PARAM_DIOD)) {
            speed = request->getParam(PARAM_DIOD)->value();
        } else {
            speed = "No message sent";
        }
        if (speed.toInt() == 1)
        {
          digitalWrite(wifi_led, HIGH);
        } 
        else
        {
          digitalWrite(wifi_led, LOW);
        }
        request->send(200, "text/plain", "Hello, GET: ");
    });
   server.onNotFound(notFound);
   server.begin();
}

void loop()
{  
    unsigned long currentMillis = millis();
    int move = digitalRead(pinPIR)
    if (move == 0){/* Отключить питание от камеры */}
    if (move == 1){/* Включи питание от камеры    */}
    if ((WiFi.status() != WL_CONNECTED) && (currentMillis - previousMillis >=interval)) {
      Serial.print(millis());
      Serial.println("Reconnecting to WiFi...");
      WiFi.disconnect();
      WiFi.reconnect();
      while (WiFi.waitForConnectResult() != WL_CONNECTED) {
        digitalWrite(wifi_led, HIGH);
        delay(500);
        digitalWrite(wifi_led, LOW);
        delay(500);
      }
      Serial.println("Connected");
      Serial.println(WiFi.localIP());
      previousMillis = currentMillis;
  }
}