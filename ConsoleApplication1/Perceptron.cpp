#include <iostream>
#include "bitmap_image.hpp"

int const MAX_ITER = 10;
float const LEARNING_RATE = 0.1;
int const NUM_INST = 200;
int const theta = 0;

int calculateOutput(int theta, float*weights, float* attributes, float bias) {
    float sum = 0;
    for (int i = 0; i < NUM_INST / 2; i++) {
         sum += attributes[i] * weights[i];
    }
    sum +=bias;
    return (sum >= theta) ? 1 : 0;
}

int main()
{
	bitmap_image image1("stone.bmp");
	bitmap_image image2("wood.bmp");
    if (!image1||!image2)
    {
        printf("Error - Failed to open: .bmp\n");
        return 1;
    }
    int outputs[NUM_INST];
    float rclass[NUM_INST][NUM_INST / 2];
    
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            rgb_t colour;
            image1.get_pixel(i, j, colour);
            rclass[i][j] = 0.3 * colour.red + 0.11 * colour.blue + 0.59 * colour.green;
         }
        outputs[i] = 0;

    }
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            rgb_t colour;
            image2.get_pixel(i, j, colour);
            rclass[100+i][j] = 0.3 * colour.red + 0.11 * colour.blue + 0.59 * colour.green;
        }
        outputs[100+i] = 1;

    }
    float weights[NUM_INST/2];// for each pixel in row
    float localError, globalError;
    int i, p, iteration, output;
    for (int i = 0; i < NUM_INST / 2; i++) {
        weights[i] = ((float)rand() / (float)RAND_MAX);// weight
    }
    
    float bias = ((float)rand() / (float)RAND_MAX);
    iteration = 0;
    do {
        iteration++;
        globalError = 0;
        // loop through all instances (complete one epoch)
        for (p = 0; p < NUM_INST; p++) {
            float row[NUM_INST/2];
            for (i = 0; i < NUM_INST/2; i++) {
                row[i] = rclass[p][i];
            }
             // calculate predicted class
            output = calculateOutput(theta, weights, row,bias);
            // difference between predicted and actual class values
            localError = outputs[p] - output;
            // update weights and bias
            for (int w = 0; w < NUM_INST / 2; w++) {
                weights[w] += LEARNING_RATE * localError * rclass[p][w];
            }
            bias += LEARNING_RATE * localError;
            // summation of squared error (error value for all instances)
            globalError += (localError * localError);
        }
        /* Root Mean Squared Error */
    std::cout<<"Iteration "<<iteration << " : RMSE = " << sqrt(globalError / double(NUM_INST)) << "\n";
} while (globalError != 0 && iteration <= MAX_ITER);

std::cout << "\n =======\nDecision boundary equation:\n";
for (int i = 0; i < NUM_INST / 2; i++) {
    std::cout << weights[i] << " * p" << i << " + ";
}
    
float unknown[NUM_INST / 2];
int r =  rand() % NUM_INST + 1;//test on training data - bad practice
for (int t = 0; t < NUM_INST / 2; t++) {
    unknown[t] = rclass[r][t];
}
    output = calculateOutput(theta, weights, unknown,bias);
    std::cout<<"\n\n =======\nRandom pixel row:";
    std::cout << "class = " << output;
    if (r < NUM_INST / 2) {
       std::cout << " when must be 0"<<std::endl;
    }
    else {
        std::cout <<" when must be 1" << std::endl;
    }

    return 0;
}
