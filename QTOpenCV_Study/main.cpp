#include "QTOpenCV_Study.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QTOpenCV_Study w;
    w.show();
	w.showMaximized();
    return a.exec();
}
