#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUMHIDDEN 2 // hidden layer 的節點數
#define NUMINPUT 2  // input 的節點數
#define NUMOUPUT 1  // output 節點數
#define NUMTRAIN 4  // 訓練資料數量

double sigmoid(double x) // sigmoid 函式
{
    double result = (1. / (1. + exp(-x)));
    return result;
}

double dsigmoid(double x) // sigmoid 的微分函式
{
    return x * (1 - x);
}

double initial_weight() // 回傳-1 ~ 1 之間的值
{
    int x = rand() % 2;
    if (!x)
    {
        return -((double)rand() / ((double)RAND_MAX));
    }
    return (double)rand() / ((double)RAND_MAX);
}

void learn(double *hidden_bias, double *hidden_weight, double *hidden, double *output_weight, int number, double *output_bias, double *inputdata, double *outputdata) // 學習過程
{
    double lr = 0.2; // learning rate

    for (int i = 0; i < NUMHIDDEN; i++) // forward propagation of hidden layer
    {
        double act1 = hidden_bias[i];
        for (int j = 0; j < NUMINPUT; j++)
        {
            act1 += *(inputdata + number * NUMINPUT + j) * *(hidden_weight + i * NUMHIDDEN + j);
        }
        hidden[i] = sigmoid(act1);
    }

    double output = 0;
    double activation = *output_bias;
    for (int i = 0; i < NUMHIDDEN; i++) // forard propagation of ouput layer
    {
        activation += hidden[i] * output_weight[i];
    }
    output = sigmoid(activation);

    double error = outputdata[number] - output; // 計算output 誤差
    double derror = error * dsigmoid(output);   //計算output delta

    double dhidden[NUMHIDDEN] = {0};
    for (int i = 0; i < NUMHIDDEN; i++) //計算hidden layer delta
    {
        dhidden[i] = output_weight[i] * derror * dsigmoid(hidden[i]);
    }

    for (int i = 0; i < NUMHIDDEN; i++) //更新output layer 權重
    {
        output_weight[i] += lr * derror * hidden[i];
    }
    *output_bias += lr * derror; //更新output layer 的bias權重

    for (int i = 0; i < NUMHIDDEN; i++) //更新hidden layer權重
    {
        for (int j = 0; j < NUMINPUT; j++)
        {
            *(hidden_weight + i * NUMHIDDEN + j) += lr * dhidden[i] * *(inputdata + number * NUMINPUT + j);
        }
    }
    for (int i = 0; i < NUMHIDDEN; i++)
    {
        hidden_bias[i] += lr * dhidden[i]; //更新hiddenlayer 的bias權重
    }
}

void initial(double *hidden_weight, double *output_weight, double *output_bias, double *hidden_bias)
{
    for (int i = 0; i < NUMHIDDEN; i++)
    {
        for (int j = 0; j < NUMINPUT; j++)
        {
            *(hidden_weight + i * NUMHIDDEN + j) = initial_weight();
        }
    }
    for (int i = 0; i < NUMHIDDEN; i++)
    {
        hidden_bias[i] = initial_weight();
        output_weight[i] = initial_weight();
    }
    *output_bias = initial_weight();
}

double predict(double *hidden_weight, double *output_weight, double *hidden, double *hidden_bias, int number, double *output_bias, double *inputdata) //結果預測
{
    for (int i = 0; i < NUMHIDDEN; i++)
    {
        double activation = hidden_bias[i];
        for (int j = 0; j < NUMINPUT; j++)
        {
            activation += *(inputdata + number * NUMINPUT + j) * *(hidden_weight + i * NUMHIDDEN + j);
        }
        hidden[i] = sigmoid(activation);
    }

    double output = 0;
    output = *output_bias;
    for (int i = 0; i < NUMHIDDEN; i++)
    {
        output += hidden[i] * output_weight[i];
    }
    output = sigmoid(output);
    return output;
}
/*
將input訓練資料設為
{0, 0, 0, 1, 1, 0, 1, 1}
*/
void setInputdata(double *inputdata)
{
    for (int index = 0; index < NUMINPUT * NUMTRAIN; index++)
    {
        if (index == 3 || index == 4 || index >= 6)
        {
            inputdata[index] = 1.;
        }
        else
        {
            inputdata[index] = 0.;
        }
    }
}
/*
將output訓練資料設為
{0, 1, 1, 0}
*/
void setOutputdata(double *outputdata)
{
    for (int index = 0; index < NUMTRAIN; index++)
    {
        if (index == 1 || index == 2)
        {
            outputdata[index] = 1.;
        }
        else
        {
            outputdata[index] = 0.;
        }
    }
}
int main()
{
    double *nothing = malloc(sizeof(double));
    double *hidden = malloc(sizeof(double) * NUMHIDDEN);                   // hidden layer值
    double *hidden_weight = malloc(sizeof(double) * NUMHIDDEN * NUMINPUT); // hidden layer 權重
    double *hidden_bias = malloc(sizeof(double) * NUMHIDDEN);              // hidden layer 的bias
    double *output_weight = malloc(sizeof(double) * NUMOUPUT * NUMHIDDEN); // output layer 的權重
    double output_bias = 0;                                                // output layer 的bias
    double *inputdata = malloc(sizeof(double) * NUMTRAIN * NUMINPUT);      //輸入的訓練資料
    double *outputdata = malloc(sizeof(double) * NUMTRAIN);                //輸出的訓練資料
    setInputdata(inputdata);
    setOutputdata(outputdata);
    srand(time(NULL));
    initial(hidden_weight, output_weight, hidden_bias, &output_bias);
    int run = 10000; //訓練10000次
    for (int i = 0; i < run; i++)
    {
        for (int index = 0; index < NUMTRAIN; index++) //輪流從第一筆資料到第四筆資料依序訓練
        {
            learn(hidden_bias, hidden_weight, hidden, output_weight, index, &output_bias, inputdata, outputdata);

            if (i % 100 == 0 && index == 0) //每訓練一百次顯示目前誤差
            {
                double error = predict(hidden_weight, output_weight, hidden, hidden_bias, index, &output_bias, inputdata) - outputdata[index];
                printf("loss = %f\n", error);
            }
        }
    }
    for (int x = 0; x < NUMTRAIN; x++) //顯示訓練結果
    {
        double value = predict(hidden_weight, output_weight, hidden, hidden_bias, x, &output_bias, inputdata);
        printf("%.0f %.0f     %f\n", *(inputdata + x * NUMINPUT + 0), *(inputdata + x * NUMINPUT + 1), value);
    }

    //釋放記憶體空間

    free(hidden);
    free(hidden_weight);
    free(hidden_bias);
    free(output_weight);
    free(inputdata);
    free(outputdata);
    return 0;
}
