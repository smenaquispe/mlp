// ProgressBar.h
#ifndef PROGRESSBAR_H
#define PROGRESSBAR_H
#include <iostream>

class ProgressBar
{
public:
    ProgressBar(int total, int width = 50) : total(total), width(width), current(0) {}

    void update(int value)
    {
        current = value;
        print();
    }

    void finish()
    {
        current = total;
        print();
        std::cout << std::endl; // New line after finishing
    }
    void reset()
    {
        current = 0;
        print();
    }
    void print()
    {
        int progress = static_cast<int>(static_cast<double>(current) / total * width);
        std::cout << "\r[";
        for (int i = 0; i < width; ++i)
        {
            if (i < progress)
                std::cout << "=";
            else if (i == progress)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << current << "/" << total << " (" << static_cast<int>(static_cast<double>(current) / total * 100) << "%)" << std::flush;
    }

private:
    int total;
    int width;
    int current;
};

#endif