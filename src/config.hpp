#pragma once

const static int HEIGHT = 720;
const static int WIDTH = 1280;

const static int N = 720 / 8;
const static float cell_size = HEIGHT / (float)N;
const static float dx = 10.0f / N;
const static int start_x = (float)WIDTH / 2.0f - N * cell_size / 2.0f;
const static int start_y = (float)HEIGHT / 2.0f - N * cell_size / 2.0f;
