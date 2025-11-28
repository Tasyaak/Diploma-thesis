/******************************************************************************
*
* Filename: FilesWork.cs
*  
* Description: 
*  Предоставляет методы для сохранения и считывания
*  массива данных разного формата в файл.
*   
******************************************************************************/

using System.Globalization;
using System.IO;
using System.Windows.Forms;
using System.Numerics;
using System.Collections.Generic;
using System;

namespace PS5000A
{
    public partial class PS5000ABlockForm : Form
    {
        /// <summary>
        /// Cохранение массива данных в файл.
        /// </summary>
        /// <param name="filename"></param>
        /// <param name="data"></param>
        private void Save2File(string filename, double[] data)
        {
            if (IsFileExist(filename) && !IsFileLocked(filename))
            {
                using (StreamWriter Writer = new StreamWriter(filename))
                {
                    string specifier = "E05";
                    CultureInfo culture = CultureInfo.CreateSpecificCulture("en-US");
                    //Writer.WriteLine(data.Length);
                    for (int index = 0; index < data.Length; index++)
                        Writer.WriteLine(data[index].ToString(specifier, culture).Replace(',', '.'));

                    Writer.Flush();
                    Writer.Close();
                }
            }
        }

        /// <summary>
        /// Перегрузка Save2File, задающая шаг записи данных.
        /// </summary>
        /// <param name="filename"></param>
        /// <param name="data"></param>
        /// <param name="step"></param>
        private void Save2File(string filename, double[] data, int step)
        {
            if (IsFileExist(filename) && !IsFileLocked(filename))
            {
                using (StreamWriter Writer = new StreamWriter(filename))
                {
                    string specifier = "E05";
                    CultureInfo culture = CultureInfo.CreateSpecificCulture("en-US");
                    //Writer.WriteLine(data.Length);
                    for (int index = 0; index < data.Length; index += step)
                        Writer.WriteLine(data[index].ToString(specifier, culture).Replace(',', '.'));

                    Writer.Flush();
                    Writer.Close();
                }
            }
        }

        /// <summary>
        /// Перегрузка Save2File, работабщая с массивом комплексных чисел и задающая шаг записи данных.
        /// </summary>
        /// <param name="filename"></param>
        /// <param name="data"></param>
        /// <param name="step"></param>
        private void Save2File(string filename, Complex[] data, int step)
        {
            if (IsFileExist(filename) && !IsFileLocked(filename))
            {
                using (StreamWriter Writer = new StreamWriter(filename))
                {
                    //Writer.WriteLine(data.Length);
                    for (int index = 0; index < data.Length; index += step)
                        Writer.WriteLine(data[index].ToString().Replace(',', '.'));

                    Writer.Flush();
                    Writer.Close();
                }
            }
        }

        /// <summary>
        /// Перегрузка Save2File, работаюащя с массивом комплексных чисел.
        /// </summary>
        /// <param name="filename"></param>
        /// <param name="data"></param>
        private void Save2File(string filename, Complex[] data)
        {
            if (IsFileExist(filename) && !IsFileLocked(filename))
            {
                using (StreamWriter Writer = new StreamWriter(filename))
                {
                    //Writer.WriteLine(data.Length);
                    for (int index = 0; index < data.Length; index++)
                        Writer.WriteLine(data[index].ToString().Replace(',', '.'));

                    Writer.Flush();
                    Writer.Close();
                }
            }
        }

        /// <summary>
        /// Перегрузка Save2File, асинхронно сохраняющая данные.
        /// </summary>
        /// <param name="filename"></param>
        /// <param name="data"></param>
        private async void Save2FileAsync(string filename, double[] data)
        {
            if (IsFileExist(filename) && !IsFileLocked(filename))
            {
                using (StreamWriter Writer = new StreamWriter(filename))
                {
                    string specifier = "E05";
                    CultureInfo culture = CultureInfo.CreateSpecificCulture("en-US");
                    //Writer.WriteLine(data.Length);
                    for (int index = 0; index < data.Length; index++)
                        await Writer.WriteLineAsync(data[index].ToString(specifier, culture).Replace(',', '.'));

                    Writer.Flush();
                    Writer.Close();
                }
            }
        }

        /// <summary>
        /// Перегрузка Save2File, асинхронно сохраняющая данные с заданным шагом.
        /// </summary>
        /// <param name="filename"></param>
        /// <param name="data"></param>
        /// <param name="step"></param>
        private async void Save2FileAsync(string filename, double[] data, int step)
        {
            if (IsFileExist(filename) && !IsFileLocked(filename))
            {
                using (StreamWriter Writer = new StreamWriter(filename))
                {
                    string specifier = "E05";
                    CultureInfo culture = CultureInfo.CreateSpecificCulture("en-US");
                    //Writer.WriteLine(data.Length);
                    for (int index = 0; index < data.Length; index += step)
                        await Writer.WriteLineAsync(data[index].ToString(specifier, culture).Replace(',', '.'));

                    Writer.Flush();
                    Writer.Close();
                }
            }
        }

        /// <summary>
        /// Cчитывает данные из файла, преобразует их в числа типа double и возвращает массив этих чисел. Длина массива - в первой строке файла.
        /// </summary>
        private double[] LoadFromFile(string filename)
        {
            if (IsFileExist(filename) && !IsFileLocked(filename))
            {
                using (StreamReader Reader = new StreamReader(filename))
                {
                    int length = int.Parse(Reader.ReadLine());
                    double[] data_list = new double[length];

                    for (int index = 0; index < length; index++)
                        data_list[index] = double.Parse(Reader.ReadLine().Replace('.', ','));

                    Reader.Close();
                    return data_list;
                }
            }
            return null;
        }

        /// <summary>
        /// Cчитывает данные измерения из файла, преобразует их в числа типа double и возвращает массив этих чисел. Длина файла заранее не известна.
        /// </summary>
        private double[] LoadMeteringFromFile(string filename)
        {
            IFormatProvider formatter = new NumberFormatInfo { NumberDecimalSeparator = "." };
            if (IsFileExist(filename) && !IsFileLocked(filename))
            {
                using (StreamReader Reader = new StreamReader(filename))
                {
                    List<double> data = new List<double>();

                    string input = null;
                    while ((input = Reader.ReadLine()) != null)
                        data.Add(double.Parse(input, formatter));

                    double[] data_list = data.ToArray();
                    return data_list;
                }
            }
            return null;
        }

        /// <summary>
        ///  Cчитывает данные из файла, преобразует их в комплексные числа и возвращает массив комплексных чисел.
        /// </summary>
        private Complex[] LoadFromFileC(string filename)
        {
            if (IsFileExist(filename) && !IsFileLocked(filename))
            {
                using (StreamReader Reader = new StreamReader(filename))
                {
                    int length = int.Parse(Reader.ReadLine());
                    Complex[] data_list = new Complex[length];

                    for (int index = 0; index < length; index++)
                        data_list[index] = StringToComplex(Reader.ReadLine());
                    //   A[i] = Complex.Parse(Reader.ReadLine().Replace('.', ','));

                    Reader.Close();
                    return data_list;
                }
            }
            return null;
        }

        /// <summary>
        /// Проверяет, существует ли файл в системе.
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        private bool IsFileExist(string filePath)
        {
            if (File.Exists(filePath))
                return true;
            else
            {
                MessageBox.Show("Ошибка! Файл по указанному пути не существует.");
                return false;
            }
        }

        /// <summary>
        /// Проверяет, не занят ли файл каким-либо другим процессом.
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        private bool IsFileLocked(string filePath)
        {
            try
            {
                using (FileStream fileStream = File.Open(filePath, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
                {
                    fileStream.Close();
                    // Файл не занят другим процессом, так как удалось открыть его с указанными параметрами.
                    return false;
                }
            }
            catch (IOException)
            {
                // Файл занят другим процессом.
                return true;
            }
        }
    }
}