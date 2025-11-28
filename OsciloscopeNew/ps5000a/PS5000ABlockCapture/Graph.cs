/******************************************************************************
*
* Filename: Graph.cs
*  
* Description:
*   Функции для работы с графиком, отображающим сигнал.
*   
******************************************************************************/

using System;
using System.Windows.Forms;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;


namespace PS5000A
{
    public partial class PS5000ABlockForm : Form
    {
        /// <summary>
        /// Отдельный класс с переменными, описывающими график
        /// </summary>
        public class GraphStatus
        {
            // График в процессе отрисовки
            public static bool visualising_now = false;

            // График перетаскивается зажатой кнопкой мыши в данный момент
            public static bool dragging_now = false;

            // Минимальное и максимальное значение точек сигнала по оси Х
            public static double minX, maxX;

            // Минимальное и максимальное ОТОБРАЖАЕМОЕ значение точек сигнала по оси Х. То есть из всего 
            // сигнала от minX до maxX отображается некоторая его часть от minDisplayedX до maxDisplayedX.
            // При этом, очевидно, minX <= minDisplayedX < maxDisplayedX <= maxX
            public static double minDisplayedX, maxDisplayedX;

            // Ширина в пикселях области между осями графика (то есть только той области, в которой происходит
            // отрисовка, не включающей подписи к осям)
            public static int innerPlotWidth;

            // Расстояние в пикселях от координатной оси Y до положения курсора мыши (отсчитываемое
            // в положительном направлении оси X) в момент наступления последнего
            // события мыши
            public static int cursorCoordX;

            // Расстояние в пикселях от координатной оси Y до пикселя, в котором была
            // зажата кнопка мыши и начато перетаскивание графика 
            public static int draggingStartedAt;

            // Минимальное и максимальное значения по оси X в момент начала перетаскивания графика
            public static double draggingStartedWithMinX, draggingStartedWithMaxX;   
        }

        /// <summary>
        /// Добавляет график в интерфейс программы
        /// </summary>
        public void InitializeChart()
        {
            this.chart1.MouseWheel += Chart1_MouseWheel;
            this.chart1.MouseDown += new System.Windows.Forms.MouseEventHandler(this.chart1_MouseDown);
            this.chart1.MouseMove += new System.Windows.Forms.MouseEventHandler(this.chart1_MouseMove);
            this.chart1.MouseUp += new System.Windows.Forms.MouseEventHandler(this.chart1_MouseUp);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="data"></param>
        private void Visualase(double[] data)
        {
            try
            {
                _oscilloscope_timestep = double.Parse(timebaseTextBox.Text);
                if (_oscilloscope_timestep < 4.0)
                    throw new Exception("Invalid timestep");

                _oscilloscope_timestep = (_oscilloscope_timestep - 3.0) / 62500000.0;
            }
            catch (Exception exception)
            {
                if (exception.Message == "Invalid timestep")
                    MessageBox.Show("Не удалось распознать шаг по времени");
            }

            // Границы сигнала по оси X
            GraphStatus.minX = 0;
            GraphStatus.maxX = _oscilloscope_timestep
                * (double.Parse(stepsAfterTextBox.Text) + double.Parse(stepsBeforeTextBox.Text));

            // Задание начальных границ отображения - сигнал отображается полностью
            GraphStatus.minDisplayedX = GraphStatus.minX;
            GraphStatus.maxDisplayedX = GraphStatus.maxX;

            PlotDataSegment(data);
        }

        /// <summary>
        /// Отображает данные на графике
        /// </summary>
        /// <param name="data"></param>
        private void PlotDataSegment(double[] data)
        {
            if (GraphStatus.visualising_now) return;
            GraphStatus.visualising_now = true;

            //Bitmap box = new Bitmap(tabControl1.TabPages[page_num].Width, tabControl1.TabPages[page_num].Height);
            //Graphics g = Graphics.FromImage(box);

            int length = data.Length;

            //Pen pp = new Pen(color);
            double max_abs = 0;

            for (int i = 0; i < length; i++)
            {
                if (max_abs < Math.Abs(data[i]))
                {
                    max_abs = Math.Abs(data[i]);
                }
            }

            if (max_abs == 0)
            {
                return;
            }

            //chart1.Hide();
            chart1.ChartAreas[0].AxisX.Minimum = GraphStatus.minDisplayedX;
            chart1.ChartAreas[0].AxisX.Maximum = GraphStatus.maxDisplayedX;
            chart1.ChartAreas[0].AxisY.Maximum = max_abs * 1000.0;// _input_ranges[comboRangeA.SelectedIndex] / 65000.0;
            chart1.ChartAreas[0].AxisY.Minimum = -max_abs * 1000.0;//  - _input_ranges[comboRangeA.SelectedIndex] / 65000.0;
            
            // Определяем шаг сетки
            chart1.ChartAreas[0].AxisX.MajorGrid.Interval = _oscilloscope_timestep * (double.Parse(stepsBeforeTextBox.Text) + double.Parse(stepsAfterTextBox.Text)) / 10.0;

            int plotWidth = chart1.Size.Width;
            // Если точек на графике слишком много, на один вертикальный ряд пикселей может приходиться их сразу
            // несколько. pointDensity - желательное предельное число точек на один вертикальный ряд пикселей, 
            // чтобы слишком много точек не тормозили отображение. Чтобы этого добиться, отображаются
            // не все точки, а подпоследовательность исходной последовательности - точки, следующие друг
            // за другом с шагом step
            const int pointDensity = 20;
            int step = 1;
            if (1.0 * length / plotWidth > pointDensity)
            {
                step = (int)1.0 * length / plotWidth / pointDensity;
            }

            // Добавление точек на график
            chart1.Series["Series1"].Points.Clear();
            for (int i = 0; i < length - 1 - step; i += step)
            {
                chart1.Series["Series1"].Points.AddXY((double)i * _oscilloscope_timestep, (double)data[i] * 1000.0);
            }

            GraphStatus.visualising_now = false;
        }
        
        private async void VisualaseAsync(double[] data)
        {
            if (!GraphStatus.visualising_now)
            {
                await Task.Run(() => Visualase(_arrA));
            }
        }

        private void Visualase(long[] data)
        {
            if (!GraphStatus.visualising_now)
            {
                double[] dadada = new double[data.Length];
                for (int i = 0; i < data.Length; i++)
                {
                    dadada[i] = (double)data[i];
                }
                Visualase(dadada);
            }
        }

        /// <summary>
        /// Обрабатывает событие прокрутки колеса мыши над графиком
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Chart1_MouseWheel(object sender, MouseEventArgs e)
        {
            // Нахождение координат мыши в момент события
            FindMouseEventCoordinates(e);

            // Масштабный коэффициент оси X
            double scale = (GraphStatus.maxDisplayedX - GraphStatus.minDisplayedX) / GraphStatus.innerPlotWidth;

            // Координата указателя мыши по оси X на графике в секундах (а не в пикселях, как 
            // у GraphStatus.cursorCoordX)
            double mouseTimeCoord = GraphStatus.minDisplayedX + scale * GraphStatus.cursorCoordX;

            // Дальнейшие действия, уменьшение или увеличение графика, зависят 
            // от того, в каком направлении прокручивается колесико мыши
            if (e.Delta > 0)
            {
                // Колесико мыши было прокручено вверх, масштаб графика увеличивается
                GraphStatus.minDisplayedX = 0.5 * (mouseTimeCoord + GraphStatus.minDisplayedX);
                GraphStatus.maxDisplayedX = 0.5 * (mouseTimeCoord + GraphStatus.maxDisplayedX);
            }
            else
            {
                // Колесико мыши было прокручено вниз, масштаб графика уменьшается
                GraphStatus.minDisplayedX = 2 * GraphStatus.minDisplayedX - mouseTimeCoord;
                GraphStatus.maxDisplayedX = 2 * GraphStatus.maxDisplayedX - mouseTimeCoord;
            }

            // График должен заполнять всю область отображения
            if (GraphStatus.minDisplayedX < GraphStatus.minX)
                GraphStatus.minDisplayedX = GraphStatus.minX;
            if (GraphStatus.maxDisplayedX > GraphStatus.maxX)
                GraphStatus.maxDisplayedX = GraphStatus.maxX;

            // Отрисовка графика на обновленном интервале
            //PlotDataSegment(data);
            chart1.ChartAreas[0].AxisX.Minimum = GraphStatus.minDisplayedX;
            chart1.ChartAreas[0].AxisX.Maximum = GraphStatus.maxDisplayedX;

            chart1.ChartAreas[0].AxisX.MajorGrid.Interval = (GraphStatus.maxDisplayedX - GraphStatus.minDisplayedX) / 10;
        }

        /// <summary>
        /// Обрабатывает событие зажатия кнопки мыши над графиком
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void chart1_MouseDown(object sender, MouseEventArgs e)
        {
            // Нахождение координат мыши в момент события
            FindMouseEventCoordinates(e);

            // Допустимы зажатия мыши только внутри координатных осей
            if (GraphStatus.cursorCoordX < 0 || GraphStatus.cursorCoordX > GraphStatus.innerPlotWidth)
                return;

            // Вход в режим перетаскивания графика
            GraphStatus.dragging_now = true;
            GraphStatus.draggingStartedAt = GraphStatus.cursorCoordX;
            GraphStatus.draggingStartedWithMinX = GraphStatus.minDisplayedX;
            GraphStatus.draggingStartedWithMaxX = GraphStatus.maxDisplayedX;
        }

        /// <summary>
        /// Обрабатывает событие перетаскивания мыши с зажатой кнопкой над графиком
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void chart1_MouseMove(object sender, MouseEventArgs e)
        {
            // Нахождение координат мыши в момент события
            FindMouseEventCoordinates(e);

            if (!GraphStatus.dragging_now) return;

            // Величина сдвига
            double shift = (GraphStatus.maxDisplayedX - GraphStatus.minDisplayedX) * (GraphStatus.cursorCoordX - GraphStatus.draggingStartedAt) / GraphStatus.innerPlotWidth;

            // График нельзя сдвигать за пределы области отображения
            if (GraphStatus.draggingStartedWithMinX - shift < GraphStatus.minX) return;
            if (GraphStatus.draggingStartedWithMaxX - shift > GraphStatus.maxX) return;

            // Выполнение сдвига графика
            GraphStatus.minDisplayedX = GraphStatus.draggingStartedWithMinX - shift;
            GraphStatus.maxDisplayedX = GraphStatus.draggingStartedWithMaxX - shift;
            chart1.ChartAreas[0].AxisX.Minimum = GraphStatus.minDisplayedX;
            chart1.ChartAreas[0].AxisX.Maximum = GraphStatus.maxDisplayedX;
        }

        /// <summary>
        /// Обрабатывает событие отпускания кнопки мыши, зажатой над графиком
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void chart1_MouseUp(object sender, MouseEventArgs e)
        {
            // Нахождение координат мыши в момент события
            FindMouseEventCoordinates(e);

            GraphStatus.dragging_now = false;
        }

        /// <summary>
        /// При возникновении события мыши над графиком сохраняет информацию о событии в объекте GraphStatus
        /// </summary>
        /// <param name="e"></param>
        private void FindMouseEventCoordinates(MouseEventArgs e)
        {
            // Расстояние от области диаграммы до границы объекта
            double chartAreaShift = chart1.Size.Width * chart1.ChartAreas[0].Position.X * 0.01;

            // Расстояние на графике от координатной оси Y до границы области диаграммы 
            double plotShift = chart1.Size.Width 
                * (chart1.ChartAreas[0].Position.Width * 0.01) 
                * (chart1.ChartAreas[0].InnerPlotPosition.X * 0.01);

            // Расстояние в пикселях от координатной оси Y до положения курсора мыши (отсчитывается
            // в положительном направлении оси X)
            GraphStatus.cursorCoordX = (int) (e.X - chartAreaShift - plotShift);

            // Текущая ширина в пикселях области между осями графика (то есть только той области, в которой происходит
            // отрисовка, не включающей подписи к осям)
            GraphStatus.innerPlotWidth = (int) (chart1.Size.Width
                * (chart1.ChartAreas[0].Position.Width * 0.01)
                * (chart1.ChartAreas[0].InnerPlotPosition.Width * 0.01));
        }
    }
}