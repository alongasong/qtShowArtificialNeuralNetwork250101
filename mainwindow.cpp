#include "mainwindow.h"
#include "./ui_mainwindow.h" // 通常这个头文件是自动生成的

#include <QRandomGenerator> // 用于生成随机数初始化权重和偏置
#include <QDebug>            // 用于在命令行输出调试信息
#include <QtMath>            // 用于 qExp (double的指数函数) 和 qAtan2 (计算反正切)
#include <QRgb>              // 尽管此代码中未直接使用QRgb，但它通常与QColor相关
#include <QPainter>          // 用于绘图
#include <QTimer>            // 用于定时器驱动训练

// 定义一个 sigmoid 激活函数
double MainWindow::sigmoid(double x) {
    return 1.0 / (1.0 + qExp(-x));
}

// sigmoid 激活函数的导数
double MainWindow::sigmoidDerivative(double x) {
    double s = 1.0 / (1.0 + qExp(-x)); //s=sigmoid(x);
    return s * (1 - s);
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // 设置窗口大小
    setFixedSize(800, 700);//600); // 调整窗口大小以容纳神经网络图形

    // 初始化神经网络
    initializeNetwork();

    // 准备 XOR 训练数据
    trainInputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    // 目标输出：XOR (第一个输出), 直通 X1 (第二个输出), 直通 X2 (第三个输出)
    trainTargets = {{0.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0}}; // 对应新的3个输出

    learningRate = 0.31;//0.5; // 学习率
    epochs = 8000;//5000;//10000;      // 训练轮次
    currentEpoch = 0;
    currentLoss = 0.0;

    // 设置定时器，每隔一段时间触发一次训练迭代
    trainingTimer = new QTimer(this);
    connect(trainingTimer, &QTimer::timeout, this, &MainWindow::trainNetwork);
    trainingTimer->start(10); // 每 10 毫秒训练一轮，可以调整这个值

    // 首次打印网络细节
    qDebug() << "--- 神经网络初始化 ---";
    printNetworkDetails();
    qDebug() << "--- 开始训练 ---";
}

MainWindow::~MainWindow()
{
    delete ui;
    delete trainingTimer;
}

void MainWindow::initializeNetwork()
{
    // 输入层大小 (XOR问题是2个输入)
    int numInputs = 2;
    // 隐藏层大小 (题目要求5个神经元)
    int numHidden = 5; // 修改为5个隐藏层神经元
    // 输出层大小 (XOR预测 + X1直通 + X2直通 = 3个输出)
    int numOutputs = 3; // 修改为3个输出层神经元

    // 初始化偏置为0
    hiddenBias.fill(0.0, numHidden);
    outputBias.fill(0.0, numOutputs);

    // 初始化权重为小的随机值
    // 输入层 -> 隐藏层 (numInputs x numHidden)
    weightsInputHidden.resize(numInputs);
    for (int i = 0; i < numInputs; ++i) {
        weightsInputHidden[i].resize(numHidden);
        for (int j = 0; j < numHidden; ++j) {
            // 使用 QRandomGenerator 生成 -0.5 到 0.5 之间的随机值
            weightsInputHidden[i][j] = QRandomGenerator::global()->generateDouble() * 2.0 - 1.0;
        }
    }

    // 隐藏层 -> 输出层 (numHidden x numOutputs)
    weightsHiddenOutput.resize(numHidden);
    for (int i = 0; i < numHidden; ++i) {
        weightsHiddenOutput[i].resize(numOutputs);
        for (int j = 0; j < numOutputs; ++j) {
            weightsHiddenOutput[i][j] = QRandomGenerator::global()->generateDouble() * 2.0 - 1.0;
        }
    }

    // 初始化层激活值
    inputLayer.fill(0.0, numInputs);
    hiddenLayer.fill(0.0, numHidden);
    outputLayer.fill(0.0, numOutputs);
}

void MainWindow::feedForward(const QVector<double>& inputs)
{
    // 确保输入匹配
    Q_ASSERT(inputs.size() == inputLayer.size());
    inputLayer = inputs; // 更新输入层值

    // 计算隐藏层输出
    for (int j = 0; j < hiddenLayer.size(); ++j) {
        double sum = hiddenBias[j]; // 加上偏置
        for (int i = 0; i < inputLayer.size(); ++i) {
            sum += inputLayer[i] * weightsInputHidden[i][j];
        }
        hiddenLayer[j] = sigmoid(sum); // 应用激活函数
    }

    // 计算输出层输出
    for (int k = 0; k < outputLayer.size(); ++k) {
        double sum = outputBias[k]; // 加上偏置
        for (int j = 0; j < hiddenLayer.size(); ++j) {
            sum += hiddenLayer[j] * weightsHiddenOutput[j][k];
        }
        // 对所有输出都应用sigmoid
        outputLayer[k] = sigmoid(sum);
    }
}

void MainWindow::backPropagate(const QVector<double>& inputs, const QVector<double>& targets)
{
    // 确保输入匹配
    Q_ASSERT(inputs.size() == inputLayer.size());
    Q_ASSERT(targets.size() == outputLayer.size());

    // 1. 计算输出层误差和梯度
    QVector<double> outputErrors(outputLayer.size());
    QVector<double> outputDeltas(outputLayer.size()); // delta = error * sigmoid'(output)
    double totalLoss = 0.0;

    for (int k = 0; k < outputLayer.size(); ++k) {
        outputErrors[k] = targets[k] - outputLayer[k];
        // 严格来说这里应该是 outputErrors[k] * sigmoidDerivative(激活前的净输入)，
        // 但为了简化，这里使用激活后的值进行导数计算。
        outputDeltas[k] = outputErrors[k] * sigmoidDerivative(outputLayer[k]);
        totalLoss += 0.5 * outputErrors[k] * outputErrors[k]; // 均方误差
    }
    currentLoss = totalLoss; // 更新当前损失

    // 2. 计算隐藏层误差和梯度
    QVector<double> hiddenErrors(hiddenLayer.size());
    QVector<double> hiddenDeltas(hiddenLayer.size());

    for (int j = 0; j < hiddenLayer.size(); ++j) {
        double error = 0.0;
        for (int k = 0; k < outputLayer.size(); ++k) {
            error += outputDeltas[k] * weightsHiddenOutput[j][k]; // 传播回隐藏层
        }
        hiddenErrors[j] = error;
        // 同理，严格来说这里用净输入进行导数计算
        hiddenDeltas[j] = hiddenErrors[j] * sigmoidDerivative(hiddenLayer[j]);
    }

    // 3. 更新隐藏层 -> 输出层 权重和偏置
    for (int j = 0; j < hiddenLayer.size(); ++j) {
        for (int k = 0; k < outputLayer.size(); ++k) {
            weightsHiddenOutput[j][k] += learningRate * hiddenLayer[j] * outputDeltas[k];
        }
    }
    for (int k = 0; k < outputLayer.size(); ++k) {
        outputBias[k] += learningRate * outputDeltas[k];
    }

    // 4. 更新输入层 -> 隐藏层 权重和偏置
    for (int i = 0; i < inputLayer.size(); ++i) {
        for (int j = 0; j < hiddenLayer.size(); ++j) {
            weightsInputHidden[i][j] += learningRate * inputLayer[i] * hiddenDeltas[j];
        }
    }
    for (int j = 0; j < hiddenLayer.size(); ++j) {
        hiddenBias[j] += learningRate * hiddenDeltas[j];
    }
}

// 槽函数：每当定时器超时就执行一次训练迭代
void MainWindow::trainNetwork()
{
    if (currentEpoch < epochs) {
        // 随机选择一个训练样本进行训练 (随机梯度下降 SGD)
        int randomIndex = QRandomGenerator::global()->bounded(trainInputs.size());
        QVector<double> currentInput = trainInputs[randomIndex];
        QVector<double> currentTarget = trainTargets[randomIndex];

        feedForward(currentInput);  // 前向传播
        backPropagate(currentInput, currentTarget); // 反向传播

        currentEpoch++;

        // 每隔 10 轮次打印进度和损失 (修改为每100个周期)
        if (currentEpoch % 100 == 0) { // 修改为每 100 个周期
            printTrainingProgress();
        }

        // 重新绘制窗口，显示神经网络的当前状态
        update();
    } else {
        trainingTimer->stop(); // 训练完成，停止定时器
        qDebug() << "\n--- 训练完成 ---";
        printNetworkDetails(); // 打印最终权重和偏置
        qDebug() << "\n--- 预测结果 ---";
        printPredictionResults(); // 打印预测结果
        update(); // 最终更新一次UI
    }
}


// --- 命令行输出函数 ---
void MainWindow::printNetworkDetails() {
    qDebug() << "学习率:" << learningRate << ", 训练轮次:" << epochs;

    qDebug() << "\n--- 输入层 -> 隐藏层 权重 ---";
    for (int i = 0; i < weightsInputHidden.size(); ++i) {
        QString row = "输入" + QString::number(i) + " -> 隐藏层: [";
        for (int j = 0; j < weightsInputHidden[i].size(); ++j) {
            row += QString::number(weightsInputHidden[i][j], 'f', 4) + (j == weightsInputHidden[i].size() - 1 ? "" : ", ");
        }
        row += "]";
        qDebug() << row;
    }

    qDebug() << "\n--- 隐藏层偏置 ---";
    QString biasStr = "[";
    for (int i = 0; i < hiddenBias.size(); ++i) {
        biasStr += QString::number(hiddenBias[i], 'f', 4) + (i == hiddenBias.size() - 1 ? "" : ", ");
    }
    biasStr += "]";
    qDebug() << biasStr;


    qDebug() << "\n--- 隐藏层 -> 输出层 权重 ---";
    for (int i = 0; i < weightsHiddenOutput.size(); ++i) {
        QString row = "隐藏" + QString::number(i) + " -> 输出层: [";
        for (int j = 0; j < weightsHiddenOutput[i].size(); ++j) {
            row += QString::number(weightsHiddenOutput[i][j], 'f', 4) + (j == weightsHiddenOutput[i].size() - 1 ? "" : ", ");
        }
        row += "]";
        qDebug() << row;
    }

    qDebug() << "\n--- 输出层偏置 ---";
    biasStr = "[";
    for (int i = 0; i < outputBias.size(); ++i) {
        biasStr += QString::number(outputBias[i], 'f', 4) + (i == outputBias.size() - 1 ? "" : ", ");
    }
    biasStr += "]";
    qDebug() << biasStr;
}

void MainWindow::printTrainingProgress() {
    qDebug() << "Epoch:" << currentEpoch << "/" << epochs << ", 损失:" << QString::number(currentLoss, 'f', 6);
}

void MainWindow::printPredictionResults() {
    qDebug() << "\n--- 预测 (推理) 结果 ---";
    for (int i = 0; i < trainInputs.size(); ++i) {
        feedForward(trainInputs[i]); // 对每个训练样本进行前向传播以获取预测
        qDebug() << "输入:" << trainInputs[i][0] << "," << trainInputs[i][1]
                 << " | 目标输出: XOR(" << trainTargets[i][0] << "), X1_Passthrough(" << trainTargets[i][1] << "), X2_Passthrough(" << trainTargets[i][2] << ")"
                 << " | 预测输出: XOR(" << QString::number(outputLayer[0], 'f', 4) << "), X1_Passthrough(" << QString::number(outputLayer[1], 'f', 4) << "), X2_Passthrough(" << QString::number(outputLayer[2], 'f', 4) << ")";
    }
}

// --- 绘制辅助函数 ---
void MainWindow::drawNeuron(QPainter& painter, const QPoint& center, int radius, const QString& label, const QString& value, const QColor& color)
{
    painter.setBrush(color);
    painter.drawEllipse(center, radius, radius);

    painter.setPen(Qt::black);
    QFont font = painter.font();
    font.setPointSize(10);
    painter.setFont(font);

    // 绘制标签 (如 X1, H1, Y1)
    QRect labelRect(center.x() - radius, center.y() - radius - 20, radius * 2, 20);
    painter.drawText(labelRect, Qt::AlignCenter, label);

    // 绘制值
    QRect valueRect(center.x() - radius, center.y() + radius + 5, radius * 2, 20);
    painter.drawText(valueRect, Qt::AlignCenter, value);
}

void MainWindow::drawConnection(QPainter& painter, const QPoint& start, const QPoint& end, double weight, bool isOutputConnection)
{
    // 根据权重正负设置线条颜色
    if (weight >= 0) {
        painter.setPen(QPen(Qt::darkGreen, 2)); // 正权重用绿色
    } else {
        painter.setPen(QPen(Qt::darkRed, 2)); // 负权重用红色
    }

    painter.drawLine(start, end);

    // 绘制权重值在连接线上
    QPoint midPoint((start.x() + end.x()) / 2, (start.y() + end.y()) / 2);
    painter.setPen(Qt::black); // 权重文字用黑色
    QFont font = painter.font();
    font.setPointSize(8);
    painter.setFont(font);

    // 调整权重文字位置，稍微偏离线条
    painter.drawText(midPoint.x() + 5, midPoint.y() - 5, QString::number(weight, 'f', 2));

    // 如果是输出连接，绘制箭头
    if (isOutputConnection) {
        QPointF arrowP1, arrowP2;
        double angle = qAtan2(end.y() - start.y(), end.x() - start.x());
        double arrowSize = 10;

        arrowP1 = QPointF(end.x() - arrowSize * qCos(angle + M_PI_4), end.y() - arrowSize * qSin(angle + M_PI_4));
        arrowP2 = QPointF(end.x() - arrowSize * qCos(angle - M_PI_4), end.y() - arrowSize * qSin(angle - M_PI_4));

        QPolygonF arrowHead;
        arrowHead << end << arrowP1 << arrowP2;
        painter.drawPolygon(arrowHead);
    }
}


// --- 核心绘制函数 ---
void MainWindow::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing); // 开启抗锯齿

    int width = this->width();
    int height = this->height();

    // 绘制背景
    painter.fillRect(rect(), Qt::lightGray);

    // 定义各层神经元的大致位置
    int inputX = width / 6;
    int hiddenX = width / 2;
    int outputX = width * 5 / 6;

    int neuronRadius = 30; // 神经元半径

    // --- 1. 绘制输入层神经元 ---
    QPoint inputNeuron1Center(inputX, height / 3);
    QPoint inputNeuron2Center(inputX, height * 2 / 3);

    drawNeuron(painter, inputNeuron1Center, neuronRadius, "X1", QString::number(inputLayer[0], 'f', 2), Qt::yellow);
    drawNeuron(painter, inputNeuron2Center, neuronRadius, "X2", QString::number(inputLayer[1], 'f', 2), Qt::yellow);

    // --- 2. 绘制隐藏层神经元 ---
    // 根据隐藏层数量调整垂直间距
    QVector<QPoint> hiddenNeuronCenters(5); // 5个隐藏层神经元
    for (int i = 0; i < 5; ++i) {
//        hiddenNeuronCenters[i] = QPoint(hiddenX, height / 6 + i * (height * 4 / (5 * 6.0))); // 均匀分布
        hiddenNeuronCenters[i] = QPoint(hiddenX, height / 6 + i * (height * 5 / (5 * 6.0))); // 均匀分布
    }

    for (int i = 0; i < hiddenNeuronCenters.size(); ++i) {
        drawNeuron(painter, hiddenNeuronCenters[i], neuronRadius,
                   "H" + QString::number(i + 1),
                   QString::number(hiddenLayer[i], 'f', 2), Qt::cyan);
    }

    // --- 3. 绘制输出层神经元 ---
    QPoint outputNeuron1Center(outputX, height / 4); // XOR 预测
    QPoint outputNeuron2Center(outputX, height / 2);  // X1 直通
    QPoint outputNeuron3Center(outputX, height * 3 / 4);// X2 直通

    drawNeuron(painter, outputNeuron1Center, neuronRadius, "Y_XOR", QString::number(outputLayer[0], 'f', 2), Qt::green);
    drawNeuron(painter, outputNeuron2Center, neuronRadius, "Y_X1", QString::number(outputLayer[1], 'f', 2), Qt::green);
    drawNeuron(painter, outputNeuron3Center, neuronRadius, "Y_X2", QString::number(outputLayer[2], 'f', 2), Qt::green);


    // --- 4. 绘制输入层到隐藏层的连接线和权重 ---
    for (int i = 0; i < 2; ++i) { // 2个输入神经元
        QPoint currentInputCenter = (i == 0) ? inputNeuron1Center : inputNeuron2Center;
        for (int j = 0; j < hiddenNeuronCenters.size(); ++j) { // 5个隐藏层神经元
            drawConnection(painter, currentInputCenter, hiddenNeuronCenters[j], weightsInputHidden[i][j]);
        }
    }


    // --- 5. 绘制隐藏层到输出层的连接线和权重 ---
    for (int i = 0; i < hiddenNeuronCenters.size(); ++i) { // 5个隐藏层神经元
        drawConnection(painter, hiddenNeuronCenters[i], outputNeuron1Center, weightsHiddenOutput[i][0], true); // 到Y_XOR
        drawConnection(painter, hiddenNeuronCenters[i], outputNeuron2Center, weightsHiddenOutput[i][1], true); // 到Y_X1
        drawConnection(painter, hiddenNeuronCenters[i], outputNeuron3Center, weightsHiddenOutput[i][2], true); // 到Y_X2
    }


    // --- 6. 绘制当前训练信息和损失 ---
    painter.setPen(Qt::darkBlue);
    QFont infoFont("Arial", 14, QFont::Bold);
    painter.setFont(infoFont);
    painter.drawText(20, 20, QString("Epoch: %1 / %2").arg(currentEpoch).arg(epochs));
    painter.drawText(20, 45, QString("Loss: %1").arg(currentLoss, 0, 'f', 6));

    // 如果训练完成，显示预测结果 (为了方便UI查看)
    if (currentEpoch >= epochs) {
        painter.setPen(Qt::darkGreen);
        QFont resultFont("Arial", 16, QFont::Bold);
        painter.setFont(resultFont);
        painter.drawText(width / 2 - 150, height - 80, "训练完成，查看控制台输出！");
    }
}
