/******************************************************************************
*
* Filename: Helpers.cs
*  
* Description:
*   Содержит разные математические функции и 
*   преобразования для работы с массивами данных double.
*   
******************************************************************************/

using System;
using System.Linq;
using System.Numerics;
using System.Windows.Forms;
using System.Text;

namespace PS5000A
{
    public partial class PS5000ABlockForm : Form
    {
        //единственная дебажная версия
        //работает

        /// <summary>
        /// Основная задача функции FuncMult заключается в умножении элементов массива f1 на значения,
        /// полученные из массива filter на основе определенных условий.
        /// </summary>
        /// <param name="f1"></param>
        /// <param name="f1dx"></param>
        /// <param name="f1x0"></param>
        /// <param name="filter"></param>
        public double[] FuncMult(double[] f1, double f1dx, double f1x0, Complex[] filter)
        {
            double[] result = new double[f1.Length];
            double mult;
            int index = 0;
            for (int integralStep = 0; integralStep < f1.Length; integralStep++)
            {
                double x = f1x0 + f1dx * (double)integralStep;
                if (x < filter[0].Real)
                {
                    mult = filter[0].Imaginary;
                }
                else
                {

                    if (x > filter[filter.Length - 1].Real)
                    {
                        mult = filter[filter.Length - 1].Imaginary;
                    }
                    else
                    {
                        while (filter[index + 1].Real < x)
                            index++;


                        double a = filter[index].Imaginary;
                        double b = (filter[index + 1].Imaginary - filter[index].Imaginary) / (filter[index + 1].Real - filter[index].Real);
                        double x_ = x - filter[index].Real;
                        mult = a + b * x_;


                        //============================================================================
                        //где нарушена логика, надо переписать
                        //============================================================================

                        //while ((filter[j].Real < x) && (j < (filter.Length - 1)))
                        //{
                        //    j++;
                        //}
                        //if (j == (filter.Length - 1))
                        //{
                        //    mult = filter[j].Real;
                        //}
                        //else
                        //{
                        //    double b = (filter[j + 1].Imaginary - filter[j].Imaginary) / (filter[j + 1].Real - filter[j].Real);
                        //    double a = filter[j].Imaginary;
                        //    double x_ = x - filter[j].Real;
                        //    mult = a + b * x_;
                        //}
                    }
                }
                result[integralStep] = mult * f1[integralStep];
            }
            return result;
        }

        /// <summary>
        /// Перегрузка функции FuncMult принимающая и возвращающая комплексное число.
        /// </summary>
        /// <param name="f1"></param>
        /// <param name="f1dx"></param>
        /// <param name="f1x0"></param>
        /// <param name="filter"></param>
        public Complex[] FuncMult(Complex[] f1, double f1dx, double f1x0, Complex[] filter)
        {
            Complex[] result = new Complex[f1.Length];
            double mult = 0;
            int index = 0;

            for (int integralStep = 0; integralStep < f1.Length; integralStep++)
            {
                double x = f1x0 + f1dx * (double)integralStep;

                if (x < filter[0].Real)
                {
                    mult = filter[0].Imaginary;
                }
                else
                {

                    if (x > filter[filter.Length - 1].Real)
                    {
                        mult = filter[filter.Length - 1].Imaginary;
                    }
                    else
                    {
                        while (filter[index + 1].Real < x)
                            index++;

                        double a = filter[index].Imaginary;
                        double b = (filter[index + 1].Imaginary - filter[index].Imaginary) / (filter[index + 1].Real - filter[index].Real);
                        double x_ = x - filter[index].Real;

                        mult = a + b * x_;
                    }
                }
                result[integralStep] = mult * f1[integralStep];
            }

            return result;
        }


        /// <summary>
        /// Преобразование Фурье для входных данных из data.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="dt"></param>
        /// <param name="t0"></param>
        /// <param name="f0"></param>
        /// <param name="df"></param>
        /// <param name="nf"></param>
        private Complex[] FurieTransf(double[] data, double dt, double t0, double f0, double df, int nf)
        {
            double w0 = 2 * M_PI * f0;
            double dw = 2 * M_PI * df;
            double mult = Math.Sqrt(1.0 / 2.0 / M_PI);
            Complex[] result = new Complex[nf];
            double w = w0;
            int length = data.Length;

            for (int currentResult = 0; currentResult < nf; currentResult++)
            {
                Complex base_exp = Complex.Exp(-1 * w * t0 * Complex.ImaginaryOne);
                Complex mult_exp = Complex.Exp(-1 * w * dt * Complex.ImaginaryOne);
                Complex exp = base_exp * mult_exp;

                result[currentResult] = data[0] * base_exp + data[length - 1] * Complex.Exp(-1 * w * (t0 + (double)(length - 1) * dt) * Complex.ImaginaryOne);
                result[currentResult] /= 2.0;

                for (int index = 1; index < length - 1; index++)
                {
                    result[currentResult] += data[index] * exp;
                    exp *= mult_exp;
                }

                result[currentResult] *= dt * mult;
                w += dw;
            }

            return result;
        }

        /// <summary>
        /// Обратное преобразование Фурье для входных данных из data.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="dt"></param>
        /// <param name="t0"></param>
        /// <param name="nt"></param>
        /// <param name="f0"></param>
        /// <param name="df"></param>
        private Complex[] FurieTransfReverse(Complex[] data, double dt, double t0, int nt, double f0, double df)
        {
            double w0 = 2 * M_PI * f0;
            double dw = 2 * M_PI * df;
            double mult = Math.Sqrt(1.0 / 2.0 / M_PI);
            Complex[] result = new Complex[nt];
            double t = t0;
            int length = data.Length;

            for (int currentResult = 0; currentResult < nt; currentResult++)
            {
                Complex base_exp = Complex.Exp(t * w0 * Complex.ImaginaryOne);
                Complex mult_exp = Complex.Exp(t * dw * Complex.ImaginaryOne);
                Complex exp = base_exp * mult_exp;

                result[currentResult] = data[0] * base_exp + data[length - 1] * Complex.Exp(t * (w0 + (double)(length - 1) * dw) * Complex.ImaginaryOne);
                result[currentResult] /= 2.0;

                for (int index = 1; index < length - 1; index++)
                {
                    result[currentResult] += data[index] * exp;
                    exp *= mult_exp;
                }

                result[currentResult] *= dw * mult;
                t += dt;

            }
            return result;
        }

        /// <summary>
        /// Возвращает среднее значение из всех элементов в data.
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        private double FindAvg(double[] data)
        {
            double average = data.Average();
            return average;
        }

        /// <summary>
        /// Возвращает среднее значение из всех элементов в data, 
        /// преобразуя их перед этим из long в double.
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        private long FindAvg(long[] data)
        {
            double average = (double)data.Average();
            return (long)average;
        }

        /// <summary>
        /// Изменяет первые index элементов массива data на среднее значение 
        /// полученное из всех элементов массива.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="index"></param>
        private void SuppressSpikes1(double[] data, int index)
        {
            double average = FindAvg(data);
            for (int currentData = 0; currentData < index; currentData++)
                data[currentData] = average;

        }

        /// <summary>
        /// Изменяет первые index элементов массива data на отрицательное среднее значение 
        /// полученное из всех элементов массива.
        /// </summary>
        /// <param name="data"></param>
        private void NoOffset1(double[] data)
        {
            double average = FindAvg(data);
            for (int currentData = 0; currentData < data.Length; currentData++)
                data[currentData] -= average;

        }

        /// <summary>
        /// Паралелльно изменяет первые index элементов массива data на среднее значение 
        /// полученное из всех элементов массива.
        /// </summary>
        /// <param name="data"></param>
        public void SuppressSpikes(double[] data)
        {
            double average = FindAvg(data);
            var newdata = data.AsParallel().Select(x => x = average);
        }

        /// <summary>
        /// Паралелльно изменяет первые index элементов массива data на
        /// отрицательное среднее значение, полученное из всех элементов массива.
        /// </summary>
        /// <param name="data"></param>
        public void NoOffset(double[] data)
        {
            double A = FindAvg(data);
            var newdata = data.AsParallel().Select(x => x -= A);
        }

        /// <summary>
        /// Этот метод выполняет операцию сглаживания массива Array_, 
        /// заменяя каждый элемент на среднее значение элементов в его окрестности в диапазоне kernel_len.
        /// </summary>
        /// <param name="Array_"></param>
        /// <param name="kernel_len"></param>
        private static void RunAvg(ref double[] Array_, int kernel_len = 10)
        {
            double[] Array_buf = new double[Array_.Length];

            for (int index = kernel_len + 1; index < Array_.Length - (kernel_len + 1); index++)
            {
                double summ = 0;

                for (int j = -kernel_len; j < kernel_len / 2; j++)
                    summ += Array_[index + j];

                Array_buf[index] = summ / (double)kernel_len;
            }

            for (int index = kernel_len + 1; index < Array_.Length - (kernel_len + 1); index++)
                Array_[index] = Array_buf[index];
        }

        /// <summary>
        /// Преобразует строку в комплексное число
        /// </summary>
        /// <param name="startString"></param>
        /// <returns></returns>
        private Complex StringToComplex(string startString)
        {
            int positon = startString.IndexOf(".0.") + 2;
            string string1 = startString.Substring(1, positon - 1).Replace('.', ',');
            string string2 = startString.Substring(positon + 1, startString.Length - positon - 3).Replace('.', ',');
            Complex complex = new Complex(double.Parse(string1), double.Parse(string2));
            return complex;

        }

        /// <summary>
        /// Структура для рабоыт с фиксированной точкой
        /// </summary>
        public struct FixedPoint
        {
            // Количество битов для дробной части
            private const int FractionalBits = 16;
            // Множитель для перевода дробных значений в FixedPoint
            private const int FractionalMultiplier = 1 << FractionalBits;

            // Значение фиксированной точки в целочисленной форме
            private int value;

            /// <summary>
            /// Конструктор, принимающий целое значение и преобразующий его в FixedPoint
            /// </summary>
            /// <param name="intValue"></param>
            public FixedPoint(int intValue)
            {
                value = intValue * FractionalMultiplier;
            }

            /// <summary>
            /// Конструктор, принимающий значение типа double и преобразующий его в FixedPoint
            /// </summary>
            /// <param name="doubleValue"></param>
            public FixedPoint(double doubleValue)
            {
                value = (int)(doubleValue * FractionalMultiplier);
            }

            // Перегрузка операторов для выполнения базовых арифметических операций между значениями с FixedPoint

            /// <summary>
            /// Переопределение оператора сложения для FixedPoint
            /// </summary>
            /// <param name="a"></param>
            /// <param name="b"></param>
            /// <returns></returns>
            public static FixedPoint operator +(FixedPoint a, FixedPoint b)
            {
                return new FixedPoint(a.value + b.value);
            }

            /// <summary>
            /// Переопределение оператора вычитания для FixedPoint
            /// </summary>
            /// <param name="a"></param>
            /// <param name="b"></param>
            /// <returns></returns>
            public static FixedPoint operator -(FixedPoint a, FixedPoint b)
            {
                return new FixedPoint(a.value - b.value);
            }

            /// <summary>
            /// Переопределение оператора умножения для FixedPoint
            /// </summary>
            /// <param name="a"></param>
            /// <param name="b"></param>
            /// <returns></returns>
            public static FixedPoint operator *(FixedPoint a, FixedPoint b)
            {
                // Для умножения двух фиксированных точек умножаем их значения и сдвигаем результат на количество битов для дробной части
                return new FixedPoint((a.value * b.value) >> FractionalBits);
            }

            /// <summary>
            /// Переопределение оператора деления для FixedPoint
            /// </summary>
            /// <param name="a"></param>
            /// <param name="b"></param>
            /// <returns></returns>
            public static FixedPoint operator /(FixedPoint a, FixedPoint b)
            {
                // Для деления двух фиксированных точек умножаем делимое на 2^16 и затем делим на делитель
                return new FixedPoint((a.value << FractionalBits) / b.value);
            }

            /// <summary>
            /// Перегрузка неявного преобразования из int в FixedPoint
            /// </summary>
            /// <param name="value"></param>
            public static implicit operator FixedPoint(int value)
            {
                return new FixedPoint(value);
            }

            /// <summary>
            /// Неявное преобразование из double в FixedPoint
            /// </summary>
            /// <param name="value"></param>
            public static implicit operator FixedPoint(double value)
            {
                return new FixedPoint(value);
            }

            /// <summary>
            /// Явное преобразование из FixedPoint в int
            /// </summary>
            /// <param name="value"></param>
            public static explicit operator int(FixedPoint value)
            {
                return value.value / FractionalMultiplier;
            }

            /// <summary>
            /// Явное преобразование из FixedPoint в double
            /// </summary>
            /// <param name="value"></param>
            public static explicit operator double(FixedPoint value)
            {
                return (double)value.value / FractionalMultiplier;
            }

            /// <summary>
            /// Явное преобразование из FixedPoint в double
            /// </summary>
            /// <param name="value"></param>
            public static explicit operator float(FixedPoint value)
            {
                return (float)value.value / FractionalMultiplier;
            }

            public static bool operator <(FixedPoint a, FixedPoint b)
            {
                return a.value < b.value;
            }

            public static bool operator >(FixedPoint a, FixedPoint b)
            {
                return a.value > b.value;
            }

            public static bool operator <=(FixedPoint a, FixedPoint b)
            {
                return a.value <= b.value;
            }

            public static bool operator >=(FixedPoint a, FixedPoint b)
            {
                return a.value >= b.value;
            }

            public static bool operator ==(FixedPoint a, FixedPoint b)
            {
                return a.value == b.value;
            }

            public static bool operator !=(FixedPoint a, FixedPoint b)
            {
                return a.value != b.value;
            }

            public override string ToString()
            {
                return ((double)value / FractionalMultiplier).ToString();
            }

        }

        struct IntFixedPoint
        {
            int _fixedPoint, _digitNumber;

            public IntFixedPoint(int value, int separator)
            {
                _fixedPoint = value;
                _digitNumber = separator;
            }

            public int fixedPoint
            {
                get { return _fixedPoint; }
                set { _fixedPoint = value; }
            }

            public int digitNumber
            {
                get { return _digitNumber; }
                set { _digitNumber = value; }
            }

            public override string ToString()
            {
                StringBuilder sb = new StringBuilder(fixedPoint.ToString());
                sb.Insert(sb.Length - digitNumber, ".");
                return sb.ToString();
            }

            public double ToDouble()
            {
                StringBuilder sb = new StringBuilder(fixedPoint.ToString());
                sb.Insert(sb.Length - digitNumber, ".");
                return double.Parse(sb.ToString());
            }

        }
    }
}