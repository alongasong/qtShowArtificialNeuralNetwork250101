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

// 定义一个 sigmoid 激活函数 (可以作为全局函数，也可以作为MainWindow成员函数)
// 注意：如果在MainWindow中定义了成员函数版本的sigmoid，这里可以移除
// 为了避免重复定义，建议移除这里的全局sigmoid函数，只保留MainWindow的成员函数版本
// 不过您的mainwindow.cpp已经正确使用了MainWindow::sigmoid，所以这里可以保留，不影响编译
// 但是，如果MainWindow::sigmoid被调用，会优先调用成员函数版本。
// 为了代码一致性，我们已将MainWindow::sigmoid作为成员函数，所以此处可选择删除，但为保持最小改动，暂时保留。
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
