#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPainter>
#include <QPaintEvent>
#include <QVector> // 用于存储权重、偏置和神经元激活值
#include <cmath>   // 用于 sigmoid 函数
#include <QTimer>  // 用于定时触发训练
#include <QPushButton> // 新增：用于按钮

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void paintEvent(QPaintEvent *event) override;
    // 重写 resizeEvent 以在窗口大小改变时重新定位按钮 (可选，但推荐)
    void resizeEvent(QResizeEvent *event) override;


private slots:
    void trainNetwork(); // 槽函数，用于触发训练
    // 新增槽函数，用于处理隐藏层神经元按钮点击
    void toggleHiddenNeuronTraining(int neuronIndex, bool freeze); // freeze 为 true 表示冻结，false 表示解冻

private:
    Ui::MainWindow *ui;

    // --- 神经网络相关数据 ---
    // 输入层 (2个神经元)
    QVector<double> inputLayer;
    // 隐藏层 (5个神经元)
    QVector<double> hiddenLayer;
    QVector<double> hiddenBias;
    QVector<QVector<double>> weightsInputHidden; // 2x5 矩阵

    // 输出层 (3个神经元)
    QVector<double> outputLayer;
    QVector<double> outputBias;
    QVector<QVector<double>> weightsHiddenOutput; // 5x3 矩阵

    // 训练数据 (XOR + X1直通 + X2直通)
    QVector<QVector<double>> trainInputs;
    QVector<QVector<double>> trainTargets;

    double learningRate;
    int    epochs;
    int    currentEpoch;
    double currentLoss; // 用于存储当前的损失

    QTimer *trainingTimer; // 定时器，用于逐步训练

    // 新增：隐藏层神经元冻结状态
    // 如果 hiddenNeuronFrozenStatus[i] 为 true，表示第 i 个隐藏层神经元被冻结，不参与训练
    QVector<bool> hiddenNeuronFrozenStatus;

    // 新增：用于隐藏层神经元的按钮
    QVector<QPushButton*> btnToggleA; // 'a' 按钮，用于冻结
    QVector<QPushButton*> btnToggleB; // 'b' 按钮，用于解冻

    // 神经网络核心函数
    void initializeNetwork();
    void feedForward(const QVector<double>& inputs);
    void backPropagate(const QVector<double>& inputs, const QVector<double>& targets);
    double sigmoid(double x); // 辅助函数
    double sigmoidDerivative(double x); // sigmoid导数

    // 绘制辅助函数
    void drawNeuron(QPainter& painter, const QPoint& center, int radius, const QString& label, const QString& value, const QColor& color = Qt::white);
    void drawConnection(QPainter& painter, const QPoint& start, const QPoint& end, double weight, bool isOutputConnection = false);

    // 辅助函数：根据神经元索引获取其中心点 (用于按钮定位)
    QPoint getHiddenNeuronCenter(int index);

    // 命令行输出函数
    void printNetworkDetails();
    void printTrainingProgress();
    void printPredictionResults();
};

#endif // MAINWINDOW_H
