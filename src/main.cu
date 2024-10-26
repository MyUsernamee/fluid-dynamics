#include "raylib.h"
#include <vector>
#include <glm/glm.hpp>
#include <algorithm>
#include <string.h>
#include <functional>
#include "simulation.hpp"
#include "config.hpp"
#include <stdio.h>
#include <iostream>

#define AUDIO_BUFFER_SIZE 1024

enum DrawMode
{

    COLOR = 0,
    PRESSURE = 1,
    VELOCITY = 2,

};

Vector2 getLocalPosition(int x, int y)
{
    return {(x - start_x) / cell_size, (y - start_y) / cell_size};
}

Vector2 getGlobalPosition(int x, int y)
{

    return {x * cell_size + start_x + cell_size / 2.0f, y * cell_size + start_y + cell_size / 2.0f};
}

void drawArrow(glm::vec2 direction, int x, int y, int w)
{
    DrawLineEx({(float)x, (float)y}, {x + direction.x, y + direction.y}, w, WHITE);
}

template <typename T>
std::pair<T, T> min_max(std::function<std::pair<bool, T>(int index)> f, int count, T min, T max)
{

    for (int i = 0; i < count; ++i)
    {
        std::pair<bool, T> value = f(i);

        if (!value.first)
            continue;

        min = std::min(min, value.second);
        max = std::max(max, value.second);
    }

    return {min, max};
}

struct Particle
{

    glm::vec2 position;
    glm::vec2 velocity;
    float life = 10.0;
};

void updateParticles(Grid<FluidData> data, std::vector<Particle> &particles, float dt)
{
    for (auto &particle : particles)
    {

        auto vel = getDataAtPoint(data, particle.position.x / dx, particle.position.y / dx);
        particle.velocity = vel.vel / vel.density;
        particle.position += particle.velocity * dt;
        particle.life -= dt;
    }

    for (int i = 0; i < particles.size(); i++)
    {
        if (particles.at(i).life <= 0)
        {
            particles.erase(particles.begin() + i);
            i--;
        }
    }
}

void draw(Grid<FluidData> data, DrawMode mode, std::vector<Particle> particles)
{
    std::pair<float, float> lim;
    float min;
    float max;
    lim = min_max<float>([&](int i)
                         { return std::make_pair(!data.data[i].wall, data.data[i].getPressure(dx * dx)); }, N * N, 99999999999999999999999.0, 0.0);

    max = lim.second;
    min = lim.first;

    if (mode == PRESSURE)
    {
        // Draw a little box to show limits
        DrawRectangleLines(10, 9, 40, 81, WHITE);
        for (int i = 0; i < 80; ++i)
        {
            DrawLine(11, i + 10, 49, i + 10, ColorFromHSV(250.0 - std::fmod((80.0 - i) / 80.0 * 250.0, 360.0), 1.0, 1.0));
        }

        DrawText(TextFormat("%f", max), 60, 9, 12, WHITE);
        DrawText(TextFormat("%f", min), 60, 81, 12, WHITE);
    }

    for (int x = 0; x < N; ++x)
    {

        // DrawLine(start_x + x * cell_size, 0, start_x + x * cell_size, HEIGHT, WHITE);

        for (int y = 0; y < N; ++y)
        {

            Color color = BLACK;

            glm::vec3 c;
            glm::vec2 v;

            switch (mode)
            {

            case PRESSURE:

                color = ColorFromHSV(std::fmod(250.0 - (data.get(x, y).getPressure(dx * dx) - min) / (max - min) * 250.0, 360.0), 1.0, 1.0);

                // We are going to draw a handy diagram for helping visualize

                break;

            case COLOR:

                c = glm::clamp(data.get(x, y).color, glm::vec3(0.0), glm::vec3(1.0));

                color = ColorFromNormalized({c.x, c.y, c.z, 1.0});

                break;

            case VELOCITY:

                v = glm::clamp(glm::abs(data.get(x, y).vel / data.get(x, y).density / 4.0f), glm::vec2(0.0), glm::vec2(1.0));
                color = ColorFromNormalized({v.x, v.y, 0.0, 1.0});

                break;
            }

            // drawArrow(data.get(x, y).vel * cell_size, start_x + x * cell_size + 0.5 * cell_size, start_y + y * cell_size + 0.5 * cell_size, 1.0);
            DrawRectangle(start_x + x * cell_size + 0.5 * cell_size, start_y + y * cell_size + 0.5 * cell_size, cell_size, cell_size,
                          color);

            if (data.get(x, y).wall)
            {
                DrawRectangle(start_x + x * cell_size + 0.5 * cell_size, start_y + y * cell_size + 0.5 * cell_size, cell_size, cell_size, RED);
            }
        }
    }

    for (auto particle : particles)
    {

        DrawCircle((int)(particle.position.x * (float)cell_size / dx) + start_x, (int)(particle.position.y * (float)cell_size / dx) + start_y, 1.0, WHITE);
    }
}

int main()
{

    InitWindow(WIDTH, HEIGHT, "Fluid Dynamics");
    InitAudioDevice();

    dim3 blocks(N / 16 + 1, N / 16 + 1);
    dim3 threads(16, 16);

    Simulation sim(10.0, N, N, BoundaryCondition::CLOSED, FluidData(ATM_PRESSURE, ATM_TEMP, dx));

    std::vector<Particle> particles;

    auto view_mode = PRESSURE;
    auto mouse_size = 10;

    while (!WindowShouldClose())
    {
        BeginDrawing();

        ClearBackground(BLACK);

        DrawFPS(10, 200);

        auto mouse_p = getLocalPosition(GetMouseX(), GetMouseY());
        auto mouse_d = GetMouseDelta();
        draw(sim.data, view_mode, particles);

        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        {
            // Make Wall
            fillCircle(sim.data, FluidData{glm::vec2(0.0), glm::vec3(0.0), true, 0.0, sim.dx}, mouse_p.x, mouse_p.y, mouse_size);
        }
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
        {
            fillCircle(sim.data, FluidData{(glm::vec2(mouse_d.x, mouse_d.y)) * 1.0f, glm::vec3(1.0, 0.0, 0.0), false, sim.data.get(mouse_p.x, mouse_p.y).getPressure(sim.dx * sim.dx) * 1.1f, sim.dx}, mouse_p.x, mouse_p.y, mouse_size);
        }
        if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE))
        {
            fillCircle(sim.data, FluidData{glm::vec2(0.0), glm::vec3(0.0), false, sim.data.get(mouse_p.x, mouse_p.y).getPressure(dx * dx) / 1.1f, sim.dx}, mouse_p.x, mouse_p.y, mouse_size);
        }

        // Just change color
        if (IsKeyPressed(KEY_R))
        {
            setCircleColor(sim.data, glm::vec3(1.0), mouse_p.x, mouse_p.y, mouse_size);
        }

        if (IsKeyDown(KEY_P))
        {
            particles.push_back({glm::vec2(mouse_p.x / (float)N * (dx * N), mouse_p.y / (float)N * (dx * N)), glm::vec2(0.0)});
        }

        drawArrow(getDataAtPoint(sim.data, mouse_p.x, mouse_p.y).vel * cell_size, GetMouseX(), GetMouseY(), 4);
        auto a = getGlobalPosition(mouse_p.x, mouse_p.y);
        DrawCircle(a.x, a.y, 4.0, RED);

        // Render rect lines to show mouse size and scale
        DrawRectangleLines(GetMouseX() - mouse_size / 2 * cell_size, GetMouseY() - mouse_size / 2 * cell_size, mouse_size * cell_size, mouse_size * cell_size, WHITE);

        // Draw text
        DrawText(TextFormat("View Mode: %d", view_mode), 10, 100, 12, WHITE);
        DrawText(TextFormat("Mouse Size: %d", mouse_size), 10, 130, 12, WHITE);

        mouse_size += GetMouseWheelMove();

        if (IsKeyPressed(KEY_SPACE))
        {
            view_mode = (DrawMode)((view_mode + 1) % 3);
        }
        sim.syncDataToGPU();

        float t1 = sim.getSafeDT(1.0 / 60.0f / 100.0) * 0.5;
        float t = 0.0;
        while (t < 1.0 / 60.0f)
        {
            float dt = max(min(t1, 1.0 / 60.0f - t), 1.0 / 60.0 / 4096.0f);
            t += dt;
            sim.stepNoSync(dt);
        }
        sim.syncDataToCPU();

        // Update particles
        updateParticles(sim.data, particles, 1.0 / 60.0f);

        EndDrawing();
    }

    CloseWindow();
}