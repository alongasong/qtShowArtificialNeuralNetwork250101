#include "mainwindow.h"
#include "./ui_mainwindow.h" // 通常这个头文件是自动生成的

#include <QRandomGenerator> // 用于生成随机数初始化权重和偏置
#include <QDebug>          // 用于在命令行输出调试信息
#include <QtMath>          // 用于 qExp (double的指数函数) 和 qAtan2 (计算反正切)
#include <QRgb>            // 尽管此代码中未直接使用QRgb，但它通常与QColor相关
#include <QPainter>        // 用于绘图
#include <QTimer>          // 用于定时器驱动训练
#include <QVBoxLayout>     // 可能会用到布局，尽管这里直接定位按钮

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
    setFixedSize(800, 990);//760);//700); // 调整窗口大小以容纳神经网络图形和按钮

    // 初始化神经网络
    initializeNetwork();

    // 准备 XOR 训练数据
    trainInputs = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    // 目标输出：XOR (第一个输出), 直通 X1 (第二个输出), 直通 X2 (第三个输出)
//    trainTargets = {{0.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0}}; // 对应新的3个输出
    // 改成2个输出, 把 y1 屏蔽!
    trainTargets = {{0.0,  0.0}, {1.0,  0.0}, {1.0,  1.0}, {0.0,  1.0}}; // 对应新的3个输出

    learningRate = 0.13;    //31; //3;//0.31; // 学习率
    epochs = 17000;//15000; //5000;//7000;//15000;//8000;       // 训练轮次
    currentEpoch = 0;
    currentLoss = 0.0;

    // 初始化隐藏层神经元冻结状态 (全部为false，即全部参与训练)
    hiddenNeuronFrozenStatus.fill(false, hiddenLayer.size());

    // 创建并初始化隐藏层神经元按钮
    // 需要确保 initializeNetwork() 已经运行，以便获取 hiddenLayer.size()
    for (int i = 0; i < hiddenLayer.size(); ++i) {
        QPushButton *btnA = new QPushButton("A", this); // 'A' 按钮表示冻结
        QPushButton *btnB = new QPushButton("B", this); // 'B' 按钮表示解冻

        btnA->setFixedSize(25, 25); // 设置按钮大小
        btnB->setFixedSize(25, 25);

        // 使用 Lambda 表达式连接信号和槽，传递神经元索引和冻结状态
        connect(btnA, &QPushButton::clicked, this, [this, i](){
            toggleHiddenNeuronTraining(i, true); // 冻结第 i 个神经元
        });
        connect(btnB, &QPushButton::clicked, this, [this, i](){
            toggleHiddenNeuronTraining(i, false); // 解冻第 i 个神经元
        });

        btnToggleA.append(btnA);
        btnToggleB.append(btnB);
    }

    // 首次设置按钮位置 (在构造函数中执行一次，并在 resizeEvent 中更新)
    // 注意：在构造函数中窗口可能还没有最终确定大小，所以这里只是一个初步位置
    // 实际的定位会在 paintEvent 和 resizeEvent 中完成
    resizeEvent(nullptr); // 模拟一次 resizeEvent 来初始化按钮位置

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
    // 确保删除所有动态创建的按钮
    for (QPushButton* btn : btnToggleA) {
        delete btn;
    }
    for (QPushButton* btn : btnToggleB) {
        delete btn;
    }
}

// 重写 resizeEvent 以在窗口大小改变时重新定位按钮
void MainWindow::resizeEvent(QResizeEvent *event)
{
    Q_UNUSED(event);
    int hiddenX = this->width() / 2; // 隐藏层神经元X坐标

    // 隐藏层神经元半径
    int neuronRadius = 30;

    // 重新定位每个隐藏层神经元旁的按钮
    for (int i = 0; i < hiddenLayer.size(); ++i) {
        QPoint center = getHiddenNeuronCenter(i);
        // 'A' 按钮放置在神经元左上方
        btnToggleA[i]->move(center.x() - neuronRadius - btnToggleA[i]->width() - 5, center.y() - 10);
        // 'B' 按钮放置在神经元右上方
        btnToggleB[i]->move(center.x() + neuronRadius + 5, center.y() - 10);
    }
}


void MainWindow::initializeNetwork()
{
    // 输入层大小 (XOR问题是2个输入)
    int numInputs = 2;
    // 隐藏层大小 (题目要求5个神经元)
    int numHidden = 5; // 修改为5个隐藏层神经元
    // 输出层大小 (XOR预测 + X1直通 + X2直通 = 3个输出)
    int numOutputs = 2;//3; // 修改为3个输出层神经元

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
        // 冻结的神经元在feedForward阶段仍然需要计算其输出，因为其输出是下一层的输入
        // 只是在反向传播时不更新其权重和偏置
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
        outputDeltas[k] = outputErrors[k] * sigmoidDerivative(outputLayer[k]);
        totalLoss += 0.5 * outputErrors[k] * outputErrors[k]; // 均方误差
    }
    currentLoss = totalLoss; // 更新当前损失

    // 2. 计算隐藏层误差和梯度
    QVector<double> hiddenErrors(hiddenLayer.size());
    QVector<double> hiddenDeltas(hiddenLayer.size());

    for (int j = 0; j < hiddenLayer.size(); ++j) {
        // 如果该隐藏层神经元被冻结，则不计算其误差和梯度，直接跳过
        if (hiddenNeuronFrozenStatus[j]) {
            hiddenErrors[j] = 0.0;
            hiddenDeltas[j] = 0.0;
            continue; // 跳过后续对该神经元的计算
        }

        double error = 0.0;
        for (int k = 0; k < outputLayer.size(); ++k) {
            error += outputDeltas[k] * weightsHiddenOutput[j][k]; // 传播回隐藏层
        }
        hiddenErrors[j] = error;
        hiddenDeltas[j] = hiddenErrors[j] * sigmoidDerivative(hiddenLayer[j]);
    }

    // 3. 更新隐藏层 -> 输出层 权重和偏置
    for (int j = 0; j < hiddenLayer.size(); ++j) {
        // 如果该隐藏层神经元被冻结，则不更新与其相关的输出层权重
        if (hiddenNeuronFrozenStatus[j]) {
            continue; // 跳过对该神经元连接的权重更新
        }
        for (int k = 0; k < outputLayer.size(); ++k) {
            weightsHiddenOutput[j][k] += learningRate * hiddenLayer[j] * outputDeltas[k];
        }
    }
    // 输出层偏置的更新与隐藏层冻结无关
    for (int k = 0; k < outputLayer.size(); ++k) {
        outputBias[k] += learningRate * outputDeltas[k];
    }

    // 4. 更新输入层 -> 隐藏层 权重和偏置
    for (int i = 0; i < inputLayer.size(); ++i) {
        for (int j = 0; j < hiddenLayer.size(); ++j) {
            // 如果该隐藏层神经元被冻结，则不更新与其相关的输入层权重
            if (hiddenNeuronFrozenStatus[j]) {
                continue; // 跳过对该神经元连接的权重更新
            }
            weightsInputHidden[i][j] += learningRate * inputLayer[i] * hiddenDeltas[j];
        }
    }
    for (int j = 0; j < hiddenLayer.size(); ++j) {
        // 如果该隐藏层神经元被冻结，则不更新其偏置
        if (hiddenNeuronFrozenStatus[j]) {
            continue; // 跳过对该神经元的偏置更新
        }
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

        // 每隔 100 轮次打印进度和损失
        if (currentEpoch % 100 == 0) {
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

// 新增槽函数：切换隐藏层神经元的训练状态 (冻结/解冻)
void MainWindow::toggleHiddenNeuronTraining(int neuronIndex, bool freeze)
{
    if (neuronIndex >= 0 && neuronIndex < hiddenNeuronFrozenStatus.size()) {
        hiddenNeuronFrozenStatus[neuronIndex] = freeze;
        //下面把 这个 冻结的 权重(到输出的)权重 给 置0(试一下)……
        weightsHiddenOutput[neuronIndex][0]=0;
        weightsHiddenOutput[neuronIndex][1]=0;
        //
        qDebug() << "隐藏层神经元 H" << neuronIndex + 1 << (freeze ? "已冻结" : "已解冻");
        // 更新按钮的显示状态 (可选，例如禁用已冻结的A按钮和已解冻的B按钮)
        // 可以在这里改变按钮颜色或文本，让用户知道当前状态
        btnToggleA[neuronIndex]->setEnabled(!freeze); // 如果冻结，A按钮禁用
        btnToggleB[neuronIndex]->setEnabled(freeze);  // 如果解冻，B按钮禁用
        update(); // 强制重绘，可能需要更新神经元颜色等视觉提示
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
        //把 目标输出 x1给 屏蔽了 , y2 是直通 x2=> y2
        qDebug() << "输入:" << trainInputs[i][0] << "," << trainInputs[i][1]
                 << " | 目标输出: XOR(" << trainTargets[i][0] << "), X2_Passthrough(" << trainTargets[i][1] << "), X2_Passthrough(" <<
            //trainTargets[i][2]<<
            ")"
                 << " | 预测输出: XOR(" << QString::number(outputLayer[0], 'f', 4) << "), X2_Passthrough(" << QString::number(outputLayer[1], 'f', 4) << "), X2_Passthrough(" ;
        qDebug() << QString::number(outputLayer[1], 'f', 4) << ")";
//        qDebug() << QString::number(outputLayer[2], 'f', 4) << ")";
    }
}

// --- 绘制辅助函数 ---
void MainWindow::drawNeuron(QPainter& painter, const QPoint& center, int radius, const QString& label, const QString& value, const QColor& color)
{
    // 根据冻结状态改变神经元颜色
    // 注意：这里需要知道当前绘制的是哪个隐藏层神经元才能判断其冻结状态
    // 由于 drawNeuron 是通用函数，我们不在其内部直接判断冻结状态来改变颜色。
    // 而是在 paintEvent 中调用 drawNeuron 时，根据 hiddenNeuronFrozenStatus 传递不同的颜色。

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

// 辅助函数：根据神经元索引获取其中心点 (用于按钮定位)
QPoint MainWindow::getHiddenNeuronCenter(int index) {
    int hiddenX = this->width() / 2;
    // 使用与 paintEvent 中相同的计算逻辑来确保位置一致性
    return QPoint(hiddenX, this->height() / 6 + index * (this->height() * 5 / (hiddenLayer.size() * 6.0)));
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
    QPoint inputNeuron1Center(inputX, height / 3 - 100);
    QPoint inputNeuron2Center(inputX, height * 2 / 3 +150);

    drawNeuron(painter, inputNeuron1Center, neuronRadius, "X1", QString::number(inputLayer[0], 'f', 2), Qt::yellow);
    drawNeuron(painter, inputNeuron2Center, neuronRadius, "X2", QString::number(inputLayer[1], 'f', 2), Qt::yellow);

    // --- 2. 绘制隐藏层神经元 ---
    QVector<QPoint> hiddenNeuronCenters(hiddenLayer.size()); // 使用 hiddenLayer.size() 确保动态适应
    for (int i = 0; i < hiddenLayer.size(); ++i) {
        // 均匀分布隐藏层神经元
        hiddenNeuronCenters[i] = QPoint(hiddenX, height / 6 + i * (height * 5 / (hiddenLayer.size() * 6.0)));

        // 根据冻结状态设置神经元颜色：冻结的用灰色，未冻结的用青色
        QColor neuronColor = hiddenNeuronFrozenStatus[i] ? Qt::gray : Qt::cyan;

        drawNeuron(painter, hiddenNeuronCenters[i], neuronRadius,
                   "H" + QString::number(i + 1),
                   QString::number(hiddenLayer[i], 'f', 2), neuronColor);
    }

    // --- 3. 绘制输出层神经元 ---
    QPoint outputNeuron1Center(outputX, (height / 4 ) - 60); // XOR 预测
//    QPoint outputNeuron2Center(outputX, height / 2);  // X1 直通
    QPoint outputNeuron3Center(outputX, (height * 3 / 4 ) +70);// X2 直通

    drawNeuron(painter, outputNeuron1Center, neuronRadius, "Y_XOR", QString::number(outputLayer[0], 'f', 2), Qt::green);
//    drawNeuron(painter, outputNeuron2Center, neuronRadius, "Y_X1", QString::number(outputLayer[1], 'f', 2), Qt::green);
    drawNeuron(painter, outputNeuron3Center, neuronRadius, "Y_X2", QString::number(outputLayer[1], 'f', 2), Qt::green);
//    drawNeuron(painter, outputNeuron3Center, neuronRadius, "Y_X2", QString::number(outputLayer[2], 'f', 2), Qt::green);


    // --- 4. 绘制输入层到隐藏层的连接线和权重 ---
    for (int i = 0; i < inputLayer.size(); ++i) { // 2个输入神经元
        QPoint currentInputCenter = (i == 0) ? inputNeuron1Center : inputNeuron2Center;
        for (int j = 0; j < hiddenNeuronCenters.size(); ++j) { // 隐藏层神经元数量
            drawConnection(painter, currentInputCenter, hiddenNeuronCenters[j], weightsInputHidden[i][j]);
        }
    }


    // --- 5. 绘制隐藏层到输出层的连接线和权重 ---
    for (int i = 0; i < hiddenNeuronCenters.size(); ++i) { // 隐藏层神经元数量
        drawConnection(painter, hiddenNeuronCenters[i], outputNeuron1Center, weightsHiddenOutput[i][0], true); // 到Y_XOR
//        drawConnection(painter, hiddenNeuronCenters[i], outputNeuron2Center, weightsHiddenOutput[i][1], true); // 到Y_X1
        drawConnection(painter, hiddenNeuronCenters[i], outputNeuron3Center, weightsHiddenOutput[i][1], true); // 到Y_X2
//        drawConnection(painter, hiddenNeuronCenters[i], outputNeuron3Center, weightsHiddenOutput[i][2], true); // 到Y_X2
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
