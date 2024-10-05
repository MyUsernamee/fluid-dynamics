#pragma once

const static int HEIGHT = 720;
const static int WIDTH = 1280;

const static int N = 720 / 4;
const static float TEMPERATURE = 293.17; // Temperature of gas in Kelvin

const static float cell_size = HEIGHT / (float)N;
const static float dx = 10.0f / N;
const static int start_x = (float)WIDTH / 2.0f - N * cell_size / 2.0f;
const static int start_y = (float)HEIGHT / 2.0f - N * cell_size / 2.0f;

#define R 8.31446261815324
#define MOLAR_WEIGHT_AIR 28.96 / 1000.0 // Molar weight of in KG

#define ATM_PRESSURE 101352.9
#define ATM_TEMP 293.17
