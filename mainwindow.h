#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPainter>
#include <QPaintEvent>
#include <QVector> // 用于存储权重、偏置和神经元激活值
#include <cmath>   // 用于 sigmoid 函数
#include <QTimer>  // 用于定时触发训练

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

private slots:
    void trainNetwork(); // 新增槽函数，用于触发训练

private:
    Ui::MainWindow *ui;

    // --- 神经网络相关数据 ---
    // 输入层 (2个神经元)
    QVector<double> inputLayer;
    // 隐藏层 (3个神经元)
    QVector<double> hiddenLayer;
    QVector<double> hiddenBias;
    QVector<QVector<double>> weightsInputHidden; // 2x3 矩阵

    // 输出层 (2个神经元)
    QVector<double> outputLayer;
    QVector<double> outputBias;
    QVector<QVector<double>> weightsHiddenOutput; // 3x2 矩阵

    // 训练数据 (XOR)
    QVector<QVector<double>> trainInputs;
    QVector<QVector<double>> trainTargets;

    double learningRate;
    int    epochs;
    int    currentEpoch;
    double currentLoss; // 用于存储当前的损失

    QTimer *trainingTimer; // 定时器，用于逐步训练

    // 神经网络核心函数
    void initializeNetwork();
    void feedForward(const QVector<double>& inputs);
    void backPropagate(const QVector<double>& inputs, const QVector<double>& targets);
    double sigmoid(double x); // 辅助函数，已经在 .cpp 中定义过了，这里作为成员函数声明
    double sigmoidDerivative(double x); // sigmoid导数

    // 绘制辅助函数
    void drawNeuron(QPainter& painter, const QPoint& center, int radius, const QString& label, const QString& value, const QColor& color = Qt::white);
    void drawConnection(QPainter& painter, const QPoint& start, const QPoint& end, double weight, bool isOutputConnection = false);

    // 命令行输出函数
    void printNetworkDetails();
    void printTrainingProgress();
    void printPredictionResults();
};

#endif // MAINWINDOW_H
