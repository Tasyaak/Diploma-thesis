/******************************************************************************
*
* Filename: DeviceControl.cs
*  
* Description:
*  Создаёт набор методов для работы с осциллографом: создание триггеров, 
*  настройка каналов, подключение к осциллографу, сбор данных из TextBox'ов.
*   
******************************************************************************/

using PicoPinnedArray;
using PS5000AImports;
using System;
using PicoStatus;
using System.Drawing;
using System.Threading;
using System.Windows.Forms;
using System.Linq;
using System.IO;

namespace PS5000A
{
    public partial class PS5000ABlockForm : Form
    {
        /// <summary>
        /// Настройка триггера для устройства с использованием заданных параметров.
        /// Метод последовательно устанавливает свойства триггера в _handle и возвращает
        /// статус. 
        /// </summary>
        /// <param name="channelProperties"></param>
        /// <param name="nChannelProperties"></param>
        /// <param name="triggerConditions"></param>
        /// <param name="nTriggerConditions"></param>
        /// <param name="directions"></param>
        /// <param name="delay"></param>
        /// <param name="auxOutputEnabled"></param>
        /// <param name="autoTriggerMs"></param>
        /// <returns></returns>
        private uint SetTrigger(Imports.TriggerChannelProperties[] channelProperties,
                                short nChannelProperties,
                                Imports.TriggerConditions[] triggerConditions,
                                short nTriggerConditions,
                                Imports.ThresholdDirection[] directions,
                                uint delay,
                                short auxOutputEnabled,
                                int autoTriggerMs)
        {

            // Флаг, последовательно провкеряющий корректность настройки триггера.
            uint status;

            status = Imports.SetTriggerChannelProperties(_handle, channelProperties, nChannelProperties, auxOutputEnabled, autoTriggerMs);

            if (status != StatusCodes.PICO_OK)
                return status;

            status = Imports.SetTriggerChannelConditions(_handle, triggerConditions, nTriggerConditions);

            if (status != StatusCodes.PICO_OK)
                return status;

            if (directions == null)
                directions = new Imports.ThresholdDirection[] { Imports.ThresholdDirection.None,
                                                                Imports.ThresholdDirection.None,
                                                                Imports.ThresholdDirection.None,
                                                                Imports.ThresholdDirection.None,
                                                                Imports.ThresholdDirection.None,
                                                                Imports.ThresholdDirection.None};

            status = Imports.SetTriggerChannelDirections(_handle,
                                                         directions[(int)Imports.Channel.ChannelA],
                                                         directions[(int)Imports.Channel.ChannelB],
                                                         directions[(int)Imports.Channel.ChannelC],
                                                         directions[(int)Imports.Channel.ChannelD],
                                                         directions[(int)Imports.Channel.External],
                                                         directions[(int)Imports.Channel.Aux]);

            if (status != StatusCodes.PICO_OK)
                return status;

            status = Imports.SetTriggerDelay(_handle, delay);

            if (status != StatusCodes.PICO_OK)
                return status;

            return status;
        }

        /// <summary>
        /// Инициирует подключение к осциллографу и устанавливает настройки канала.
        /// </summary>
        /// <param name="sampleCountBefore"></param>
        /// <param name="sampleCountAfter"></param>
        /// <param name="write_every"></param>
        private void set_oscilloscope(uint sampleCountBefore = 50000, uint sampleCountAfter = 50000, int write_every = 100)
        {
            // Настройка канала осциллографа
            const short enable = 1;
            const uint delay = 0;
            const short threshold = 20000;
            const short auto = 22222;

            uint all = sampleCountAfter + sampleCountBefore;
            uint status;
            int max_samples;

            status = Imports.MemorySegments(_handle, 1, out max_samples);
            Imports.Range IR = (Imports.Range)comboRangeA.SelectedIndex;
            status = Imports.SetChannel(_handle, Imports.Channel.ChannelA, 1, Imports.Coupling.PS5000A_AC, IR, 0);

            // Voltage_Range = 200;
            // status = Imports.SetChannel(_handle, Imports.Channel.ChannelA, 1, Imports.Coupling.PS5000A_AC, Imports.Range.Range_200mV, 0);
            // status = Imports.SetChannel(_handle, Imports.Channel.ChannelA, 1, Imports.Coupling.PS5000A_DC, Imports.Range.Range_200mV, 0);

            if (passBandLimit20CheckBox.Checked)
                status = Imports.SetBandwidthFilter(_handle, Imports.Channel.ChannelA, Imports.BandwidthLimiter.PS5000A_BW_20MHZ);

            status = Imports.SetSimpleTrigger(_handle, enable, Imports.Channel.External, threshold, Imports.ThresholdDirection.Rising, delay, auto);
            _ready = false;
            _callbackDelegate = BlockCallback;
            _channel_count = 1;

            // Работа с буферами
            PinnedArray<short>[] minPinned = new PinnedArray<short>[_channel_count];
            PinnedArray<short>[] maxPinned = new PinnedArray<short>[_channel_count];

            int timeIndisposed;
            _min_buffers_a = new short[all];
            _max_buffers_a = new short[all];
            minPinned[0] = new PinnedArray<short>(_min_buffers_a);
            maxPinned[0] = new PinnedArray<short>(_max_buffers_a);

            status = Imports.SetDataBuffers(_handle,
                                            Imports.Channel.ChannelA,
                                            _max_buffers_a,
                                            _min_buffers_a,
                                            (int)sampleCountAfter + (int)sampleCountBefore,
                                            0,
                                            Imports.RatioMode.None);


            _ready = false;
            _callbackDelegate = BlockCallback;

            bool retry;

            do
            {
                retry = false;
                status = Imports.RunBlock(_handle,
                                         (int)sampleCountBefore,
                                         (int)sampleCountAfter,
                                         uint.Parse(this.timebaseTextBox.Text),
                                         out timeIndisposed,
                                         0,
                                         _callbackDelegate,
                                         IntPtr.Zero);

                if (status == (short)StatusCodes.PICO_POWER_SUPPLY_CONNECTED ||
                    status == (short)StatusCodes.PICO_POWER_SUPPLY_NOT_CONNECTED ||
                    status == (short)StatusCodes.PICO_POWER_SUPPLY_UNDERVOLTAGE)
                {
                    status = Imports.ChangePowerSource(_handle, status);
                    retry = true;
                }
                else
                {
                    //  textMessage.AppendText("Run Block Called\n");
                }
            }
            while (retry);

            while (!_ready)
            {
                //    Thread.Sleep(1);
                //Application.DoEvents();///тормоза
                Thread.Sleep(0);
            }

            Imports.Stop(_handle);
            if (_ready)
            {
                short overflow;
                status = Imports.GetValues(_handle, 0, ref all, 1, Imports.RatioMode.None, 0, out overflow);

                if (status == (short)StatusCodes.PICO_OK)
                    for (int index = 0; index < all; index++)
                        masA[index] += _max_buffers_a[index] + _min_buffers_a[index];//=========================================================!
            }

            Imports.Stop(_handle);
            foreach (PinnedArray<short> pin in minPinned)
                if (pin != null)
                    pin.Dispose();

            foreach (PinnedArray<short> pin in maxPinned)
                if (pin != null)
                    pin.Dispose(); // Можно заменить проверку на одну строку: pin?.Dispose();

            // Application.DoEvents(); ///тормоза
        }

        private void setup_oscilloscope(
            ref PinnedArray<short>[] minPinned,
            ref PinnedArray<short>[] maxPinned,
            uint sampleCountBefore = 50000,
            uint sampleCountAfter = 50000,
            int write_every = 100)
        {
            // Настройка канала осциллографа
            const short enable = 1;
            const uint delay = 0;
            const short threshold = 20000;
            const short auto = 22222;

            uint all = sampleCountAfter + sampleCountBefore;
            uint status;
            int max_samples;

            status = Imports.MemorySegments(_handle, 1, out max_samples);
            Imports.Range IR = (Imports.Range)comboRangeA.SelectedIndex;
            status = Imports.SetChannel(_handle, Imports.Channel.ChannelA, 1,
                Imports.Coupling.PS5000A_AC, IR, 0);

            // Voltage_Range = 200;
            // status = Imports.SetChannel(_handle, Imports.Channel.ChannelA, 1,
            // Imports.Coupling.PS5000A_AC, Imports.Range.Range_200mV, 0);
            // status = Imports.SetChannel(_handle, Imports.Channel.ChannelA, 1,
            // Imports.Coupling.PS5000A_DC, Imports.Range.Range_200mV, 0);

            if (passBandLimit20CheckBox.Checked)
                status = Imports.SetBandwidthFilter(_handle, Imports.Channel.ChannelA,
                    Imports.BandwidthLimiter.PS5000A_BW_20MHZ);

            status = Imports.SetSimpleTrigger(_handle, enable, Imports.Channel.External,
                threshold, Imports.ThresholdDirection.Rising, delay, auto);
            _ready = false;
            _callbackDelegate = BlockCallback;
            _channel_count = 1;

            // Работа с буферами
            if (minPinned.Count() != _channel_count)
            {
                minPinned = new PinnedArray<short>[_channel_count];
                _min_buffers_a = new short[all];
                minPinned[0] = new PinnedArray<short>(_min_buffers_a);
            }

            if (maxPinned.Count() != _channel_count)
            {
                maxPinned = new PinnedArray<short>[_channel_count];
                _max_buffers_a = new short[all];
                maxPinned[0] = new PinnedArray<short>(_max_buffers_a);
            }
              
            int timeIndisposed;

            status = Imports.SetDataBuffers(_handle,
                                            Imports.Channel.ChannelA,
                                            _max_buffers_a,
                                            _min_buffers_a,
                                            (int)sampleCountAfter + (int)sampleCountBefore,
                                            0,
                                            Imports.RatioMode.None);

            _ready = false;
            _callbackDelegate = BlockCallback; 
        }

        private void start_measurements(uint sampleCountBefore = 50000, uint sampleCountAfter = 50000, int write_every = 100)
        {
            bool retry;

            uint all = sampleCountAfter + sampleCountBefore;
            uint status;
            int timeIndisposed;
            do
            {
                retry = false;
                status = Imports.RunBlock(_handle,
                                         (int)sampleCountBefore,
                                         (int)sampleCountAfter,
                                         uint.Parse(this.timebaseTextBox.Text),
                                         out timeIndisposed,
                                         0,
                                         _callbackDelegate,
                                         IntPtr.Zero);

                if (status == (short)StatusCodes.PICO_POWER_SUPPLY_CONNECTED ||
                    status == (short)StatusCodes.PICO_POWER_SUPPLY_NOT_CONNECTED ||
                    status == (short)StatusCodes.PICO_POWER_SUPPLY_UNDERVOLTAGE)
                {
                    status = Imports.ChangePowerSource(_handle, status);
                    retry = true;
                }
                else
                {
                    //  textMessage.AppendText("Run Block Called\n");
                }
            }
            while (retry);

            while (!_ready)
            {
                //    Thread.Sleep(1);
                //Application.DoEvents();///тормоза
                Thread.Sleep(0);
            }

            Imports.Stop(_handle);
            if (_ready)
            {
                short overflow;
                status = Imports.GetValues(_handle, 0, ref all, 1, Imports.RatioMode.None, 0, out overflow);

                if (status == (short)StatusCodes.PICO_OK)
                    for (int index = 0; index < all; index++)
                        masA[index] += _max_buffers_a[index] + _min_buffers_a[index];//=========================================================!
            }
        }

        private void delete_pinned(ref PinnedArray<short>[] minPinned, ref PinnedArray<short>[] maxPinned)
        {
            Imports.Stop(_handle);
            foreach (PinnedArray<short> pin in minPinned)
                if (pin != null)
                    pin.Dispose();

            foreach (PinnedArray<short> pin in maxPinned)
                if (pin != null)
                    pin.Dispose(); // Можно заменить проверку на одну строку: pin?.Dispose();

            minPinned = null;
            maxPinned = null;
        }

        /// <summary>
        /// Считывает значения из текстовых полей 
        /// и преобразует их в соответствующие числовые типы. 
        /// Затем запускает метод визаулизации.
        /// </summary>
        public void CollectData(
            uint samples1,
            uint samples2,
            uint count_avg,
            ushort RANGE_,
            long[] masA,
            double[] _arrA,
            int _all)
        {
            //uint samples1 = uint.Parse(stepsBeforeTextBox.Text);
            //uint samples2 = uint.Parse(stepsAfterTextBox.Text);
            //uint count_avg = uint.Parse(averagingsTextBox.Text);

            //masA = new long[samples1 + samples2];
            //_arrA = new double[samples1 + samples2];
            //_all = int.Parse(averagingsTextBox.Text);

            //ushort RANGE_ = _input_ranges[comboRangeA.SelectedIndex];
            double mult = 1.0 / 2.0 / (double)count_avg * RANGE_ / 65536.0 / _save;


            _stop_flag = false;

            for (uint j = 0; j < samples1 + samples2; j++)
            {
                _arrA[j] = 0;
            }

            for (uint index = 0; index < count_avg; index++)
            {
                set_oscilloscope(samples1, samples2, 1);

                if (_stop_flag)
                    break;

                _save = (int)index + 1;
                bool update = false;

                if (((index % 50) == 1) && update)
                    Application.DoEvents();

                if (visualizationCheckBox.Checked)
                {
                    if ((index % 100) == 0)
                    {
                        for (uint j = 0; j < samples1 + samples2; j++)
                        {
                            if (_stop_flag)
                                break;
                            _arrA[j] = _arrA[j] + (double)masA[j];  
                        }
                        bool _visualising_now = false;
                        if (!_visualising_now)// _
                        {
                            //Visualase(Color.Blue, arrA, 1);
                            //VisualaseAsync(Color.Blue, _arrA, 1);
                            VisualaseAsync(  _arrA );
                        }
                    }
                }

            }

            for (uint index = 0; index < samples1 + samples2; index++)
                _arrA[index] = (double)masA[index] * mult;

        }

        public void CollectData_V2(
            uint samples1,
            uint samples2,
            uint count_avg,
            ushort RANGE_,
            long[] masA,
            double[] _arrA,
            int _all)
        {

            double mult = 1.0 / 2.0 / (double)count_avg * RANGE_ / 65536.0 / _save;

            _stop_flag = false;
            PinnedArray<short>[] minPinned = null;
            PinnedArray<short>[] maxPinned = null;
            setup_oscilloscope(ref minPinned , ref maxPinned , samples1, samples2, 1);


            for (uint j = 0; j < samples1 + samples2; j++)
            {
                _arrA[j] = 0;
            }

            for (uint index = 0; index < count_avg; index++)
            {
                start_measurements(samples1, samples2, 1);

                if (_stop_flag)
                    break;

                _save = (int)index + 1;
                bool update = false;

                if (((index % 50) == 1) && update)
                    Application.DoEvents();

                if (visualizationCheckBox.Checked)
                {
                    if ((index % 100) == 0)
                    {
                        for (uint j = 0; j < samples1 + samples2; j++)
                        {
                            if (_stop_flag)
                                break;

                            _arrA[j] = _arrA[j]+ (double)masA[j];
                            //if (checkBox2.Checked)
                            //{
                            //    //      SuppressSpikes(arrA, int.Parse(textBox14.Text));
                            //}

                        }
                        bool _visualising_now = false;

                        if (!_visualising_now)// _
                        {
                            //Visualase(Color.Blue, arrA, 1);
                            VisualaseAsync(  _arrA );
                        }
                    }
                }

            }

            for (uint index = 0; index < samples1 + samples2; index++)
                _arrA[index] = (double)masA[index] * mult;

            delete_pinned(ref minPinned, ref maxPinned);
        }

        //[System.Runtime.InteropServices.DllImport("C:/Users/user1/source/repos/owon_signalProcessing/x64/Release/owon_signalProcessing.dll")]
        //[System.Runtime.InteropServices.DllImport("C:/Users/user1/source/repos/OsciloscopeNew-main/owon_signalProcessing/x64/Release/owon_signalProcessing.dll")]
        //[System.Runtime.InteropServices.DllImport("E:/OsciloscopeNew-main/owon_signalProcessing/x64/Release/owon_signalProcessing.dll")]
        [System.Runtime.InteropServices.DllImport("C:/OWON/OsciloscopeNew-Owouwun-patch-1/owon_signalProcessing/x64/Release/owon_signalProcessing.dll")]
        public static extern int getOWONData2(
            int sample_size,
            double sample_step,
            int sample_perSec_max,
            double voltage_max,
            int trigger_level,
            int reading_number,
            int offset_size,
            double[] result_arr,
            char[] result_fileName
        );

        public void CollectData_OWON()
        {
            string fileName = filePathTextBox.Text + fileNameTextBox.Text + '\0';
            char[] read_fileName = new char[fileName.Length];
            read_fileName = fileName.ToCharArray();
            double maxVoltage;
            switch (comboRangeA.SelectedIndex)
            {
                case 0:
                    maxVoltage = 10;
                    break;
                case 1:
                    maxVoltage = 20;
                    break;
                case 2:
                    maxVoltage = 50;
                    break;
                case 3:
                    maxVoltage = 100;
                    break;
                case 4:
                    maxVoltage = 200;
                    break;
                case 5:
                    maxVoltage = 500;
                    break;
                case 6:
                    maxVoltage = 1000;
                    break;
                case 7:
                    maxVoltage = 2000;
                    break;
                case 8:
                    maxVoltage = 5000;
                    break;
                default:
                    maxVoltage = 5000;
                    string message = "OWON не поддерживает вертикальное разрешение больше 5V. Продолжить с разрешением 5V?";
                    string caption = "Предупреждение";
                    MessageBoxButtons buttons = MessageBoxButtons.YesNo;
                    DialogResult result;

                    // Displays the MessageBox.
                    result = MessageBox.Show(message, caption, buttons);
                    if (result == System.Windows.Forms.DialogResult.Yes)
                    {
                        maxVoltage = 5000;
                    }
                    else
                    {
                        this.Close(); // Здесь должна быть нормальная остановка процесса считывания
                        return;
                    }
                    break;
            }
            maxVoltage /= 1000;

            Application.DoEvents();
            _arrA = new double[int.Parse(stepsAfterTextBox.Text)];
            int a = getOWONData2(
                int.Parse(stepsAfterTextBox.Text),
                (int.Parse(timebaseTextBox.Text) - 1) * 16E-9,
                125000000,
                maxVoltage,
                0,
                int.Parse(averagingsTextBox.Text),
                int.Parse(stepsBeforeTextBox.Text),
                _arrA, // Не работает. Нужно как-то отправить double-указатель на начало этого массива, тогда следующий код не будет нужен
                read_fileName
                );

            Application.DoEvents();

            string tempName = filePathTextBox.Text;
            tempName += fileNameTextBox.Text;
            using (TextReader reader = File.OpenText(tempName))
            {
                string temp;
                if (Convert.ToChar(Thread.CurrentThread.CurrentCulture.NumberFormat.NumberDecimalSeparator) == '.')
                {
                    for (int i = 0; i < int.Parse(stepsAfterTextBox.Text.Trim()); i++)
                    {
                        temp = reader.ReadLine();
                        _arrA[i] = Convert.ToDouble(temp);
                    }
                }
                else
                {
                    for (int i = 0; i < int.Parse(stepsAfterTextBox.Text.Trim()); i++)
                    {
                        temp = reader.ReadLine();
                        _arrA[i] = Convert.ToDouble(temp.Replace(".", ","));
                    }
                }
            }
        }
    }
}