/******************************************************************************
*
* Filename: Filter.cs
*  
* Description: 
*  Внутри этого класса определены несколько методов, 
*  связанных с обработкой и сохранением фильтров.
*   
******************************************************************************/

using System;
using System.IO;
using System.Globalization;
using System.Windows.Forms;
using System.Drawing;
using System.Numerics;
using System.Collections.Generic;

namespace PS5000A
{
    public partial class PS5000ABlockForm : Form
    {
        // Отдельная запись в списке истории применения фильтров к данным
        struct FilteringHistoryEntry
        {
            public string Label;
            public double[] Data;
        }

        // Список для сохранения истории применения фильтров к данным
        private List<FilteringHistoryEntry> _filtering_history;

        public class Prompt : IDisposable
        {
            private Form prompt { get; set; }
            public string Result { get; }

            public Prompt(string text, string caption)
            {
                Result = ShowDialog(text, caption);
            }
            //use a using statement
            private string ShowDialog(string text, string caption)
            {
                prompt = new Form()
                {
                    Width = 500,
                    Height = 150,
                    FormBorderStyle = FormBorderStyle.FixedDialog,
                    Text = caption,
                    StartPosition = FormStartPosition.CenterScreen,
                    TopMost = true
                };
                Label textLabel = new Label() { Left = 50, Top = 20, Text = text, Dock = DockStyle.Top, TextAlign = ContentAlignment.MiddleCenter };
                TextBox textBox = new TextBox() { Left = 50, Top = 50, Width = 400 };
                Button confirmation = new Button() { Text = "Ok", Left = 350, Width = 100, Top = 70, DialogResult = DialogResult.OK };
                confirmation.Click += (sender, e) => { prompt.Close(); };
                prompt.Controls.Add(textBox);
                prompt.Controls.Add(confirmation);
                prompt.Controls.Add(textLabel);
                prompt.AcceptButton = confirmation;

                return prompt.ShowDialog() == DialogResult.OK ? textBox.Text : "";
            }

            public void Dispose()
            {
                if (prompt != null)
                {
                    prompt.Dispose();
                }
            }
        }

        /// <summary>
        /// Сохраняет буфер комплексных чисел с заданными аргументами в файл.
        /// </summary>
        /// <param name="filename"></param>
        /// <param name="filter"></param>
        /// <param name="x_arg"></param>
        private void SaveFilter(string filename, double[] filter, double[] x_arg)
        { 
            Complex[] buf = new Complex[x_arg.Length];

            for (int index = 0; index < x_arg.Length; index++)
                buf[index] = new Complex(x_arg[index], filter[index]);

            Save2File(filename, buf);
        }

        /// <summary>
        /// Загружает фильтр из файла.
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        private Complex[] LoadFilter(string filename)
        {
            return LoadFromFileC(filename);
        }

        /// <summary>
        /// Возвращает из массива комплексных чисел действительную часть.
        /// </summary>
        /// <param name="f"></param>
        /// <returns></returns>
        private double[] ExtractFilterArgs(Complex[] f)
        {
            double[] real_numbers = new double[f.Length]; 

            for (int index = 0; index < f.Length; index++)
                real_numbers[index] = f[index].Real;
            
            return real_numbers;
        }

        /// <summary>
        /// Возвращает из массива комплексных чисел мнимую часть.
        /// </summary>
        /// <param name="f"></param>
        /// <returns></returns>
        private double[] ExtractFilterVals(Complex[] f)
        {
            double[] imaginary_numbers = new double[f.Length];

            for (int index = 0; index < f.Length; index++)
                imaginary_numbers[index] = f[index].Imaginary;
            
            return imaginary_numbers;
        }
        
        /// <summary>
        /// Выполняет сортировку значений в массивах x и f. 
        /// Используется алгоритм сортировки выбором, чтобы упорядочить элементы по возрастанию значения f.
        /// </summary>
        private void SortFilterPoints(ref double[] x, ref double[] f)
        {
            int length = x.Length;
            for (int point = 0; point < length - 1; point++)
            {
                double x_min = x[point];
                int index = point;

                for (int j = point + 1; j < length; j++)
                    if (x_min > f[j])
                    {
                        x_min = f[j];
                        index = j;
                    }
                
                if (index != point)
                {
                    double x_ = x[index];
                    double f_ = f[index];
                    x[index] = x[point];
                    f[index] = f[point];
                    x[point] = x_;
                    f[point] = f_;
                }
            }
        }

        /// <summary>
        /// Осуществляет проверку, стоит ли string только из цифр.
        /// </summary>
        /// <param name="text"></param>
        /// <returns></returns>
        private bool IsNumeric(string text)
        {
            foreach (char c in text)
            {
                if (!char.IsDigit(c))
                {
                    return false;
                }
            }
            return true;
        }

        /* 
         * Загрузка массива данных измерения из файла
         */
        private void LoadMetering()
        {
            // Имя файла с данными
            var fileName = string.Empty;
            // Окно выбора файла для загрузки данных
            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog.Title = "Load Metering Data";
                openFileDialog.CheckFileExists = true;
                openFileDialog.CheckPathExists = true;

                if (openFileDialog.ShowDialog() == DialogResult.Cancel)
                    return;

                fileName = openFileDialog.FileName;
            }

            // Чтение данных из файла
            _arrA = LoadMeteringFromFile(fileName);

            // Очистка истории применения фильтров
            ClearFilteringHistory();

            Visualase(_arrA);
        }

        /*
         * Сохранение массива данных измерения в файле
         */
        private void SaveMetering()
        {
            if (_arrA == null)
            {
                MessageBox.Show("Нет данных для сохранения!");
                return;
            }

            var fileName = string.Empty;
            using (SaveFileDialog saveFileDialog = new SaveFileDialog())
            {
                saveFileDialog.Title = "Save Metering Data";
                saveFileDialog.CheckPathExists = true;

                if (saveFileDialog.ShowDialog() == DialogResult.Cancel)
                    return;

                fileName = saveFileDialog.FileName;
            }

            // Запись данных в файл с именем fileName
            using (StreamWriter Writer = new StreamWriter(fileName))
            {
                string specifier = "E05";
                CultureInfo culture = CultureInfo.CreateSpecificCulture("en-US");
                for (int index = 0; index < _arrA.Length; index++)
                    Writer.WriteLine(_arrA[index].ToString(specifier, culture).Replace(',', '.'));

                Writer.Flush();
                Writer.Close();
            }
        }

        /*
         * Обработчик нажатия кнопки "Применить фильтр"
         */
        private void ApplyFilter()
        {
            if (filtersComboBox.SelectedItem == null) return;

            if (_arrA == null)
            {
                MessageBox.Show("Не загружены данные!");
                return;
            }

            switch (filtersComboBox.SelectedItem.ToString())
            {
                case "Среднее арифметическое":
                    ArithmeticMeanFilter();
                    break;
                case "Медианный фильтр":
                    MedianFilter();
                    break;
                case "Экспоненциальное бегущее среднее":
                    ExponentialMovingAverageFilter();
                    break;
                case "Фурье-удаление низких частот":
                    DeleteLowFreqFilter();
                    break;
                case "Фурье-удаление высоких частот":
                    DeleteHighFreqFilter();
                    break;
            }
        }

        /*
         * Обработчик нажатия кнопки "Отменить последний фильтр"
         */
        private void UndoLastFilter()
        {
            if (_filtering_history.Count == 0) return;

            // Помещение значения массива с данными из последнего элемента истории
            // применения фильтров в текущее значение _arrA
            _arrA = _filtering_history[_filtering_history.Count - 1].Data;

            // Удаление последней записи из истории фильтров
            _filtering_history.RemoveAt(_filtering_history.Count - 1);

            // Вывод результатов в истории фильтрации
            DisplayFilteringHistory();

            Visualase(_arrA);
        }

        /*
         * Очистка истории применения фильтров
         */
        private void ClearFilteringHistory()
        {
            _filtering_history = new List<FilteringHistoryEntry>();
            DisplayFilteringHistory();
        }

        /* 
         * Отображение истории применения фильтров
         */
        private void DisplayFilteringHistory()
        {
            string txt = "";
            int i = 1;
            foreach (var item in _filtering_history)
            {
                txt += i + ") " + item.Label + "\r\n";
                i++;
            }
            filtersHistoryTextBox.Text = txt;
        }

        /*
         * Функция выполняется в конце работы каждого фильтра. Здесь серия
         * команд, вынесенная в отдельную функцию, чтобы она не дублировалась
         */
        private void FinishApplyingFilter(double[] result, string description)
        {
            // Помещение текущего значения массива с данными _arrA в историю
            // применения фильтров, потом присвоение полученного после применения
            // фильтра массива переменной _arrA
            FilteringHistoryEntry oldData = new FilteringHistoryEntry();
            oldData.Label = description;
            oldData.Data = _arrA;
            _filtering_history.Add(oldData);
            _arrA = result;

            // Вывод результатов в истории фильтрации
            DisplayFilteringHistory();

            Visualase(_arrA);
        }

        /*
         * Фильтр арифметического среднего
         */
        private void ArithmeticMeanFilter()
        {
            // Размер буфера - число предыдущих значений, по которым
            // вычисляется среднее арифметическое
            int bufferSize = 0;
            string promptValue;

            // Запрос размера буфера в диалоговом окне
            using (Prompt prompt = new Prompt(
                "Введите число значений, по которым вычисляется среднее арифметическое",
                "Размер буфера"))
            {
                promptValue = prompt.Result;
            }

            // Преобразование введенной пользователем строки в число
            bool isNumber = int.TryParse(promptValue, out bufferSize);
            if (!isNumber || bufferSize < 2)
            {
                MessageBox.Show("Неверное значение!");
                return;
            }

            // Если bufferSize введен верно, применяется фильтр
            filtersHistoryTextBox.Text = "Calculating...";

            // Буфер значений
            Queue<double> buffer = new Queue<double>();
            for (int i = 1; i <= bufferSize; i++)
            {
                buffer.Enqueue(_arrA[0]);
            }

            // Массив для результата применения фильтра
            double[] result = new double[_arrA.Length];

            for (int i = 0; i < result.Length; i++)
            {
                buffer.Dequeue();
                buffer.Enqueue(_arrA[i]);

                // Сумма элементов буфера
                double sum = 0;
                foreach (var item in buffer) { sum += item; }

                result[i] = sum / bufferSize;
            }

            FinishApplyingFilter(result, "Арифметическое среднее. Размер буфера: " + bufferSize);
        }

        /*
         * Медианный фильтр
         */
        private void MedianFilter()
        {
            // Размер буфера - число значений, по которым
            // вычисляется медиана
            int bufferSize = 0;
            string promptValue;

            // Запрос размера буфера в диалоговом окне
            using (Prompt prompt = new Prompt(
                "Введите число значений, по которым вычисляется медиана",
                "Размер буфера"))
            {
                promptValue = prompt.Result;
            }

            // Преобразование введенной пользователем строки в число
            bool isNumber = int.TryParse(promptValue, out bufferSize);
            if (!isNumber || bufferSize < 3)
            {
                MessageBox.Show("Неверное значение!");
                return;
            }

            // Если bufferSize введен верно, применяется фильтр
            filtersHistoryTextBox.Text = "Calculating...";

            // Буфер значений
            List<double> buffer = new List<double>();

            // Массив для результата применения фильтра
            double[] result = new double[_arrA.Length];

            for (int i = 0; i < result.Length; i++)
            {
                // Буфер берется симметричным относительно текущего элемента
                // массива. То есть первый элемент буфера имеет индекс
                // i - (bufferSize / 2)
                int start_index = i - (bufferSize / 2);

                // Заполнение буфера
                buffer.Clear();
                for (int j = start_index; j < start_index + bufferSize; j++)
                {
                    // Если буфер выходит за границы массива значений, то 
                    // дублируются граничные значения
                    if (j < 0)
                    {
                        buffer.Add(_arrA[0]);
                    }
                    else if (j >= result.Length)
                    {
                        buffer.Add(_arrA[result.Length - 1]);
                    }
                    else
                    {
                        buffer.Add(_arrA[j]);
                    }
                }

                buffer.Sort();

                // При нечетном размере буфера берется средний элемент, при
                // четном - полусумма средних
                if (bufferSize % 2 == 1)
                {
                    result[i] = buffer[bufferSize / 2];
                }
                else
                {
                    result[i] = (buffer[bufferSize / 2 - 1] + buffer[bufferSize / 2]) / 2;
                }
            }

            FinishApplyingFilter(result, "Медианный фильтр. Размер буфера: " + bufferSize);
        }

        /*
         * Фильтр экспоненциального бегущего среднего
         */
        private void ExponentialMovingAverageFilter()
        {
            // Весовой коэффициент k - основной параметр вычисления экспоненциального
            // бегущего среднего. X_new[i] = k * X_old[i] + (1 - k) * X_new[i - 1]
            double k = 1;
            string promptValue;

            // Запрос значения k в диалоговом окне
            using (Prompt prompt = new Prompt(
                "Введите параметр k экспоненциального бегущего среднего\n X_new[i] = k * X_old[i] + (1 - k) * X_new[i - 1]",
                "Значение k"))
            {
                promptValue = prompt.Result.Replace(',', '.');
            }

            // Преобразование введенной пользователем строки в число
            NumberStyles style = NumberStyles.AllowDecimalPoint | NumberStyles.AllowExponent;
            IFormatProvider formatter = new NumberFormatInfo { NumberDecimalSeparator = "." };
            bool isNumber = double.TryParse(promptValue, style, formatter, out k);

            if (!isNumber || k < 0 || k > 1)
            {
                MessageBox.Show("Неверное значение!");
                return;
            }

            // Если k введен верно, применяется фильтр
            filtersHistoryTextBox.Text = "Calculating...";

            // Массив для результата применения фильтра
            double[] result = new double[_arrA.Length];
            result[0] = _arrA[0];

            for (int i = 1; i < result.Length; i++)
            {
                result[i] = k * _arrA[i] + (1 - k) * result[i - 1];
            }

            FinishApplyingFilter(result, "Экспоненциальное бегущее среднее. Коэффициент k = " + k);
        }

        /*
         * Фильтр, выполняющий преобразование Фурье, удаляющий низкие частоты
         * из спектра, затем выполняющий обратное преобразование Фурье.
         */
        private void DeleteLowFreqFilter()
        {
            // Частота minFreq - граничная частота, ниже которой удаляются все частоты
            // в Фурье-преобразовании сигнала
            double minFreq = -1;
            string promptValue;

            // Запрос значения minFreq в диалоговом окне
            using (Prompt prompt = new Prompt(
                "Введите минимальную сохраняемую частоту. Все частоты ниже нее будут удалены из спектра",
                "Значение частоты"))
            {
                promptValue = prompt.Result.Replace(',', '.');
            }

            // Преобразование введенной пользователем строки в число
            NumberStyles style = NumberStyles.AllowDecimalPoint | NumberStyles.AllowExponent;
            IFormatProvider formatter = new NumberFormatInfo { NumberDecimalSeparator = "." };
            bool isNumber = double.TryParse(promptValue, style, formatter, out minFreq);

            if (!isNumber || minFreq < 0)
            {
                MessageBox.Show("Неверное значение!");
                return;
            }

            // Если minFreq введена верно, применяется фильтр
            filtersHistoryTextBox.Text = "Calculating...";

            // Массив для результата Фурье-преобразования
            Complex[] spectrum = new Complex[_arrA.Length];

            // Массив для результата применения фильтра
            double[] result = new double[_arrA.Length];

            spectrum = FurieTransf(_arrA,
                _oscilloscope_timestep,
                -_oscilloscope_timestep * double.Parse(stepsBeforeTextBox.Text),
                double.Parse(FTBottomTextBox.Text),
                double.Parse(FTStepTextBox.Text),
                int.Parse(FTStepsNumberTextBox.Text));

            // Проход по спектру и удаление заданных частот
            double dFreq = double.Parse(FTStepTextBox.Text);
            double freq = double.Parse(FTBottomTextBox.Text);
            for (int i = 0; i < spectrum.Length; i++)
            {
                if (freq < minFreq) spectrum[i] = 0;
                freq += dFreq;
            }

            Complex[] restored = FurieTransfReverse(spectrum,
                _oscilloscope_timestep,
                -_oscilloscope_timestep * double.Parse(stepsBeforeTextBox.Text),
                _arrA.Length,
                double.Parse(FTBottomTextBox.Text),
                double.Parse(FTStepTextBox.Text));

            for (int k1 = 0; k1 < restored.Length; k1++)
            {
                result[k1] = restored[k1].Real * 2;//важно делать умножение на 2 так как интеграл по полубесконечному промежутку
            }

            FinishApplyingFilter(result, "Фурье-удаление низких частот. Минимальная сохраняемая частота: " + minFreq);
        }

        /*
         * Фильтр, выполняющий преобразование Фурье, удаляющий высокие частоты
         * из спектра, затем выполняющий обратное преобразование Фурье.
         */
        private void DeleteHighFreqFilter()
        {
            // Частота maxFreq - граничная частота, выше которой удаляются все частоты
            // в Фурье-преобразовании сигнала
            double maxFreq = -1;
            string promptValue;

            // Запрос значения maxFreq в диалоговом окне
            using (Prompt prompt = new Prompt(
                "Введите максимальную сохраняемую частоту. Все частоты выше нее будут удалены из спектра",
                "Значение частоты"))
            {
                promptValue = prompt.Result.Replace(',', '.');
            }

            // Преобразование введенной пользователем строки в число
            NumberStyles style = NumberStyles.AllowDecimalPoint | NumberStyles.AllowExponent;
            IFormatProvider formatter = new NumberFormatInfo { NumberDecimalSeparator = "." };
            bool isNumber = double.TryParse(promptValue, style, formatter, out maxFreq);

            if (!isNumber || maxFreq < 0)
            {
                MessageBox.Show("Неверное значение!");
                return;
            }

            // Если minFreq введена верно, применяется фильтр
            filtersHistoryTextBox.Text = "Calculating...";

            // Массив для результата Фурье-преобразования
            Complex[] spectrum = new Complex[_arrA.Length];

            // Массив для результата применения фильтра
            double[] result = new double[_arrA.Length];

            spectrum = FurieTransf(_arrA,
                _oscilloscope_timestep,
                -_oscilloscope_timestep * double.Parse(stepsBeforeTextBox.Text),
                double.Parse(FTBottomTextBox.Text),
                double.Parse(FTStepTextBox.Text),
                int.Parse(FTStepsNumberTextBox.Text));

            // Проход по спектру и удаление заданных частот
            double dFreq = double.Parse(FTStepTextBox.Text);
            double freq = double.Parse(FTBottomTextBox.Text);
            for (int i = 0; i < spectrum.Length; i++)
            {
                if (freq > maxFreq) spectrum[i] = 0;
                freq += dFreq;
            }

            Complex[] restored = FurieTransfReverse(spectrum,
                _oscilloscope_timestep,
                -_oscilloscope_timestep * double.Parse(stepsBeforeTextBox.Text),
                _arrA.Length,
                double.Parse(FTBottomTextBox.Text),
                double.Parse(FTStepTextBox.Text));

            for (int k1 = 0; k1 < restored.Length; k1++)
            {
                result[k1] = restored[k1].Real * 2;//важно делать умножение на 2 так как интеграл по полубесконечному промежутку
            }

            FinishApplyingFilter(result, "Фурье-удаление высоких частот. Максимальная сохраняемая частота: " + maxFreq);
        }



    }
}