#include "mainwindow.h"

#include <QPainter> // 包含了绘图工具
#include <QPen>     // 包含了画笔工具
#include <QBrush>   // 包含了画刷工具
#include <QFont>    // 包含了字体工具
#include <QDebug>   // 用于调试输出

// 包含了数学函数，例如 exp 用于 sigmoid
#include <cmath>

#include <QApplication>
#include <QLocale>
#include <QTranslator>

// 定义一个 sigmoid 激活函数
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QTranslator translator;
    const QStringList uiLanguages = QLocale::system().uiLanguages();
    for (const QString &locale : uiLanguages) {
        const QString baseName = "qtTestuntitled2401_" + QLocale(locale).name();
        if (translator.load(":/i18n/" + baseName)) {
            a.installTranslator(&translator);
            break;
        }
    }
    MainWindow w;
    w.show();
    return a.exec();
}


/*
void MainWindow::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event); // 告诉编译器我们没用到 event 参数，避免警告

    // 1. 创建一个 QPainter 对象，指定在当前窗口 (this) 上下文进行绘制
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing); // 开启抗锯齿，让图形更平滑

    // 2. 定义绘图参数
    int width = this->width();    // 获取窗口宽度
    int height = this->height();  // 获取窗口高度
    QPoint center(width / 2, height / 2); // 窗口中心点
    int neuronRadius = 50; // 神经元的半径

    // 3. 绘制神经元主体（一个圆）
    QPen pen(Qt::black); // 创建一个黑色的画笔
    pen.setWidth(2);      // 设置画笔粗细为 2 像素
    painter.setPen(pen);

    QBrush brush(QColor("#a2d2ff")); // 创建一个淡蓝色的画刷
    painter.setBrush(brush);

    painter.drawEllipse(center, neuronRadius, neuronRadius);

    // 4. 绘制输入连接线和输入值
    QPoint input1_start(center.x() - 200, center.y() - 80);
    QPoint input2_start(center.x() - 200, center.y());
    QPoint input3_start(center.x() - 200, center.y() + 80);

    painter.setBrush(Qt::NoBrush); // 后续只画线，不需要填充
    painter.drawLine(input1_start, center);
    painter.drawLine(input2_start, center);
    painter.drawLine(input3_start, center);

    // 5. 在神经元内部绘制激活函数符号 (例如 Σ)
    QFont font("Arial", 24, QFont::Bold); // 创建字体
    painter.setFont(font);
    painter.drawText(center.x() - 15, center.y() + 10, "Σ");

    // 6. 标注输入和权重
    font.setPointSize(12);
    font.setBold(false);
    painter.setFont(font);

    double x1 = 0.8;
    double x2 = 0.5;
    double x3 = 0.2;

    double w1 = 0.7;
    double w2 = 0.3;
    double w3 = 0.9;

    QString QSx1="x1=%.1f"+QString::number(x1);
    QString QSx2="x1=%.1f"+QString::number(x2);
    QString QSx3="x1=%.1f"+QString::number(x3);
    painter.drawText(input1_start.x() - 40, input1_start.y() + 5, QSx1);//QString("x1=%.1f", qsx1));
    painter.drawText(input2_start.x() - 40, input2_start.y() + 5, QSx2);//QString("x2=%.1f", x2));
    painter.drawText(input3_start.x() - 40, input3_start.y() + 5, QSx3);//QString("x3=%.1f", x3));

    // 标注权重
    QString QSw1="x1=%.1f"+QString::number(w1);
    QString QSw2="x1=%.1f"+QString::number(w2);
    QString QSw3="x1=%.1f"+QString::number(w3);
    painter.drawText(center.x() - 120, center.y() - 50, QSw1);//QString("w1=%.1f", w1));
    painter.drawText(center.x() - 120, center.y() + 5, QSw2);//QString("w2=%.1f", w2));
    painter.drawText(center.x() - 120, center.y() + 50, QSw3);//QString("w3=%.1f", w3));

    // 7. 绘制输出线
    QPoint output_end(center.x() + 200, center.y());
    painter.drawLine(center, output_end);
    painter.drawText(output_end.x() + 10, output_end.y() + 5, "Output");

    // 8. 显示运算过程和输出数值
    font.setPointSize(10); // 调整字体大小以便显示更多信息
    painter.setFont(font);

    // 运算公式
    painter.drawText(center.x() - 100, center.y() - 120, "Formula: Output = Sigmoid(Σ(xi * wi))");

    // 各个输入与权重的乘积
    double term1 = x1 * w1;
    double term2 = x2 * w2;
    double term3 = x3 * w3;

    QString QSterm1= "x1 * w1 = %.1f * %.1f = %.2f"+ QString::number(x1)+QString::number( w1)+QString::number( term1);
    QString QSterm2= "x1 * w1 = %.1f * %.1f = %.2f"+ QString::number(x2)+QString::number( w2)+QString::number( term2);
  QString QSterm3= "x1 * w1 = %.1f * %.1f = %.2f"+ QString::number(x3)+QString::number( w3)+QString::number( term3);
    painter.drawText(center.x() - 100, center.y() - 90,  QSterm1 );//"x1 * w1 = %.1f * %.1f = %.2f", x1, w1, term1));
    painter.drawText(center.x() - 100, center.y() - 70, QSterm2);//QString("x2 * w2 = %.1f * %.1f = %.2f", x2, w2, term2));
    painter.drawText(center.x() - 100, center.y() - 50, QSterm3);//QString("x3 * w3 = %.1f * %.1f = %.2f", x3, w3, term3));

    // 求和
    double sum = term1 + term2 + term3;
    QString QSsum= "Sum = %.2f + %.2f + %.2f = %.2f" + QString::number(term1)+ QString::number(term2)+QString::number(term3)+ "sum="+ QString::number(sum);
    painter.drawText(center.x() - 100, center.y() - 20, QSsum);//QString("Sum = %.2f + %.2f + %.2f = %.2f", term1, term2, term3, sum));

    // 应用激活函数（Sigmoid）
    double output_value = sigmoid(sum);
    QString QSoutput= QString::number(output_value);
    painter.drawText(output_end.x() + 10, output_end.y() + 25,  QSoutput);//QString("Value: %.3f", output_value));
}
*/
