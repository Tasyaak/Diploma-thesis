/******************************************************************************
*
* Filename: PS5000ABlockForm.cs
*  
* Description:
*   Файл содержит обработчиков событий для кнопок, и инииализурет начальную 
*   графический интерфейс приложения.
*   
******************************************************************************/

using PicoStatus;
using PS5000AImports;
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Ports;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using static PS5000AImports.Imports;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace PS5000A
{
    public partial class PS5000ABlockForm : Form
    {
        private FileMFT fileMFT;
        private Switch Switch1 = null;
        private Imports.ps5000aBlockReady _callbackDelegate;

        public const int BUFFER_SIZE = 1024;
        public const int MAX_CHANNELS = 4;
        public const int QUAD_SCOPE = 4;
        public const int DUAL_SCOPE = 2;
        private const uint _n = 10000;

        private const double M_PI = 3.1415926535897932384626433832795;
        private const double dt_ = 104 * 1.0E-9;

        private const string CODES = "ABCDEFGHIKJLMNOPQRSTUVWXYZ0123456789";

        private ushort[] _input_ranges = { 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000 };
        private short[] _min_buffers_a;
        private short[] _max_buffers_a;
        private short _handle;

        public uint samples1;
        public uint samples2;
        public uint count_avg;
        //private long[] masA;
        public long[] masA;
        public ushort RANGE_;


        private int _all = 0;
        private int _save = 0;
        private int _channel_count;

        private float _cnc_x = 0;
        private float _cnc_y = 0;

        private double _oscilloscope_timestep = 0;
        private double[] _arrA;

        private bool _stop_flag = false;
        private bool _switch_connected = false;
        private bool _ready = false;

        private string[] names_;
        /*
         * частоты от 0 Гц 10Mhz
         * dt =10^-7
         * df = 100;
         */

        /// <summary>
        /// Конструктор формы, инициализирующий его работу.
        /// </summary>
        public PS5000ABlockForm()
        {
            InitializeComponent();
            InitializeChart();

            oscilloscopeSwitch.DataSource = new System.Collections.Generic.List<String>
            {
                "Picoscope",
                "OWON"
            };

            comboRangeA.DataSource = System.Enum.GetValues(typeof(Imports.Range));
            progressBar1.Text = "Готов к работе";
            timer1.Interval = 300;
            timer1.Tick += new EventHandler(Timer1_Tick);
        }

        /// <summary>
        /// Функция обновляющая состояние progressBar'а.
        /// </summary>
        /// <param name="Sender"></param>
        /// <param name="e"></param>
        private void Timer1_Tick(object Sender, EventArgs e)
        {
            if (_all!=0)
            {
                progressBar1.Value = (int)(((double)_save) / _all * progressBar1.Maximum); 
            }
            else
            {
                progressBar1.Value = 0;
            }
            Refresh();
        }

        /// <summary>
        /// Проверяет, было ли установлено соединение с осциллографом.
        /// Если да - устанавливает флаг ready на true.
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="status"></param>
        /// <param name="pVoid"></param>
        private void BlockCallback(short handle, short status, IntPtr pVoid)
        {
            // flag to say done reading data
            if (status != (short)StatusCodes.PICO_CANCELLED)
                _ready = true;
        }

        private void ConnectButton_Click(object sender, EventArgs e)
        {

            try
            {
                _oscilloscope_timestep = double.Parse(timebaseTextBox.Text);
                if (_oscilloscope_timestep < 4.0)
                    throw new Exception("Invalid timestep");

                _oscilloscope_timestep = (_oscilloscope_timestep - 3.0) / 62500000.0;

                StringBuilder UnitInfo = new StringBuilder(80);

                short handle;
                string[] description = {
                           "Driver Version    ",
                           "USB Version       ",
                           "Hardware Version  ",
                           "Variant Info      ",
                           "Serial            ",
                           "Cal Date          ",
                           "Kernel Ver        ",
                           "Digital Hardware  ",
                           "Analogue Hardware "
                         };

                Imports.DeviceResolution resolution = Imports.DeviceResolution.PS5000A_DR_16BIT;

                if (_handle > 0)
                {
                    Imports.CloseUnit(_handle);
                    _handle = 0;
                    connectButton.Text = "Open Unit";
                }
                else
                {
                    uint status = Imports.OpenUnit(out handle, null, resolution);

                    if (handle > 0)
                    {
                        _handle = handle;

                        if (status == StatusCodes.PICO_POWER_SUPPLY_NOT_CONNECTED || status == StatusCodes.PICO_USB3_0_DEVICE_NON_USB3_0_PORT)
                        {
                            status = Imports.ChangePowerSource(_handle, status);
                        }
                        else if (status != StatusCodes.PICO_OK)
                        {
                            MessageBox.Show("Cannot open device error code: " + status.ToString(),
                                            "Error Opening Device",
                                            MessageBoxButtons.OK,
                                            MessageBoxIcon.Error);
                            this.Close();
                        }
                        connectButton.Text = "Close Unit";
                    }
                }
            }
            catch (Exception exception)
            {
                if (exception.Message == "Invalid timestep")
                    MessageBox.Show("Не удалось распознать шаг по времени");
            }
        }


        //Самый времяемкий метод
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea1 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend1 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series1 = new System.Windows.Forms.DataVisualization.Charting.Series();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.oscilloscopeSwitch = new System.Windows.Forms.ComboBox();
            this.label26 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.comboRangeA = new System.Windows.Forms.ComboBox();
            this.timebaseTextBox = new System.Windows.Forms.TextBox();
            this.label13 = new System.Windows.Forms.Label();
            this.connectButton = new System.Windows.Forms.Button();
            this.tabPage3 = new System.Windows.Forms.TabPage();
            this.disconnectAllButton = new System.Windows.Forms.Button();
            this.label30 = new System.Windows.Forms.Label();
            this.layersTextBox = new System.Windows.Forms.TextBox();
            this.invertSensorsButton = new System.Windows.Forms.Button();
            this.sourcesCheckedListBox = new System.Windows.Forms.CheckedListBox();
            this.receiversCheckedListBox = new System.Windows.Forms.CheckedListBox();
            this.textBoxUnitInfo = new System.Windows.Forms.TextBox();
            this.portListBox = new System.Windows.Forms.ListBox();
            this.connectPortButton = new System.Windows.Forms.Button();
            this.getPortsButton = new System.Windows.Forms.Button();
            this.label16 = new System.Windows.Forms.Label();
            this.sensorTextBox = new System.Windows.Forms.TextBox();
            this.label12 = new System.Windows.Forms.Label();
            this.label11 = new System.Windows.Forms.Label();
            this.label10 = new System.Windows.Forms.Label();
            this.disconnectButton = new System.Windows.Forms.Button();
            this.label9 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.connectAsSourceButton = new System.Windows.Forms.Button();
            this.connectAsReceiverButton = new System.Windows.Forms.Button();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.selectFolderButton = new System.Windows.Forms.Button();
            this.label28 = new System.Windows.Forms.Label();
            this.fileNameTextBox = new System.Windows.Forms.TextBox();
            this.saveEveryTextBox = new System.Windows.Forms.TextBox();
            this.label27 = new System.Windows.Forms.Label();
            this.saveRawCheckBox = new System.Windows.Forms.CheckBox();
            this.saveTimesCheckBox = new System.Windows.Forms.CheckBox();
            this.FTStepsNumberTextBox = new System.Windows.Forms.TextBox();
            this.FTStepTextBox = new System.Windows.Forms.TextBox();
            this.FTBottomTextBox = new System.Windows.Forms.TextBox();
            this.label22 = new System.Windows.Forms.Label();
            this.label19 = new System.Windows.Forms.Label();
            this.label18 = new System.Windows.Forms.Label();
            this.computeModuleCheckBox = new System.Windows.Forms.CheckBox();
            this.computeFTCheckBox = new System.Windows.Forms.CheckBox();
            this.label17 = new System.Windows.Forms.Label();
            this.filePathTextBox = new System.Windows.Forms.TextBox();
            this.suppressStartCheckBox = new System.Windows.Forms.CheckBox();
            this.autocollectButton = new System.Windows.Forms.Button();
            this.visualizationCheckBox = new System.Windows.Forms.CheckBox();
            this.stopButton = new System.Windows.Forms.Button();
            this.passBandLimit20CheckBox = new System.Windows.Forms.CheckBox();
            this.suppressedTextBox = new System.Windows.Forms.TextBox();
            this.label21 = new System.Windows.Forms.Label();
            this.stepsBeforeTextBox = new System.Windows.Forms.TextBox();
            this.label20 = new System.Windows.Forms.Label();
            this.label15 = new System.Windows.Forms.Label();
            this.averagingsTextBox = new System.Windows.Forms.TextBox();
            this.stepsAfterTextBox = new System.Windows.Forms.TextBox();
            this.label14 = new System.Windows.Forms.Label();
            this.button1 = new System.Windows.Forms.Button();
            this.collectDataButton = new System.Windows.Forms.Button();
            this.tabPage4 = new System.Windows.Forms.TabPage();
            this.useFreqFilterTextBox = new System.Windows.Forms.TextBox();
            this.useFreqFilterCheckBox = new System.Windows.Forms.CheckBox();
            this.useTimeFilterTextBox = new System.Windows.Forms.TextBox();
            this.useTimeFilterCheckBox = new System.Windows.Forms.CheckBox();
            this.autoEliminateAvgCheckBox = new System.Windows.Forms.CheckBox();
            this.applyAutoRunAvgCheckBox = new System.Windows.Forms.CheckBox();
            this.eliminateAvgButton = new System.Windows.Forms.Button();
            this.runAvgTextBox = new System.Windows.Forms.TextBox();
            this.applyRunAvgButton = new System.Windows.Forms.Button();
            this.tabPage5 = new System.Windows.Forms.TabPage();
            this.applyFreqFilterCheckBox = new System.Windows.Forms.CheckBox();
            this.suppressToTextBox = new System.Windows.Forms.TextBox();
            this.suppressToCheckBox = new System.Windows.Forms.CheckBox();
            this.differenceInfoTextBox = new System.Windows.Forms.TextBox();
            this.diffFolderTextBox = new System.Windows.Forms.TextBox();
            this.label25 = new System.Windows.Forms.Label();
            this.normBeforeOutputCheckBox = new System.Windows.Forms.CheckBox();
            this.buildDifferencesButton = new System.Windows.Forms.Button();
            this.defectFolderTextBox = new System.Windows.Forms.TextBox();
            this.label24 = new System.Windows.Forms.Label();
            this.label23 = new System.Windows.Forms.Label();
            this.noDefectFolderTextBox = new System.Windows.Forms.TextBox();
            this.tabPage6 = new System.Windows.Forms.TabPage();
            this.flowLayoutPanel1 = new System.Windows.Forms.FlowLayoutPanel();
            this.loadMeteringButton = new System.Windows.Forms.Button();
            this.saveMeteringButton = new System.Windows.Forms.Button();
            this.applyFilterButton = new System.Windows.Forms.Button();
            this.filtersComboBox = new System.Windows.Forms.ComboBox();
            this.undoLastFilterButton = new System.Windows.Forms.Button();
            this.filtersHistoryTextBox = new System.Windows.Forms.TextBox();
            this.tabPage7 = new System.Windows.Forms.TabPage();
            this.chart1 = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.tabPage8 = new System.Windows.Forms.TabPage();
            this.button15 = new System.Windows.Forms.Button();
            this.tabPage9 = new System.Windows.Forms.TabPage();
            this.normalizeCheckBox = new System.Windows.Forms.CheckBox();
            this.computeFTButton = new System.Windows.Forms.Button();
            this.label29 = new System.Windows.Forms.Label();
            this.measurementFolderTextBox = new System.Windows.Forms.TextBox();
            this.tabPage10 = new System.Windows.Forms.TabPage();
            this.rightButton = new System.Windows.Forms.Button();
            this.leftButton = new System.Windows.Forms.Button();
            this.downButton = new System.Windows.Forms.Button();
            this.upButton = new System.Windows.Forms.Button();
            this.dyTextBox = new System.Windows.Forms.TextBox();
            this.label35 = new System.Windows.Forms.Label();
            this.dxTextBox = new System.Windows.Forms.TextBox();
            this.label36 = new System.Windows.Forms.Label();
            this.label34 = new System.Windows.Forms.Label();
            this.resultsTextBox = new System.Windows.Forms.TextBox();
            this.send2TextBox = new System.Windows.Forms.TextBox();
            this.send2Button = new System.Windows.Forms.Button();
            this.port2ListBox = new System.Windows.Forms.ListBox();
            this.connectPort2Button = new System.Windows.Forms.Button();
            this.getPorts2Button = new System.Windows.Forms.Button();
            this.yTextBox = new System.Windows.Forms.TextBox();
            this.label33 = new System.Windows.Forms.Label();
            this.xTextBox = new System.Windows.Forms.TextBox();
            this.label32 = new System.Windows.Forms.Label();
            this.modeTextBox = new System.Windows.Forms.TextBox();
            this.label31 = new System.Windows.Forms.Label();
            this.send1Button = new System.Windows.Forms.Button();
            this.tabPage11 = new System.Windows.Forms.TabPage();
            this.moveAlongYCheckBox = new System.Windows.Forms.CheckBox();
            this.moveAlongXCheckBox = new System.Windows.Forms.CheckBox();
            this.fTextBox = new System.Windows.Forms.TextBox();
            this.label45 = new System.Windows.Forms.Label();
            this.offYTextBox = new System.Windows.Forms.TextBox();
            this.label43 = new System.Windows.Forms.Label();
            this.offXTextBox = new System.Windows.Forms.TextBox();
            this.label44 = new System.Windows.Forms.Label();
            this.measureAndSaveCheckBox = new System.Windows.Forms.CheckBox();
            this.button2 = new System.Windows.Forms.Button();
            this.collectData2Button = new System.Windows.Forms.Button();
            this.nyTextBox = new System.Windows.Forms.TextBox();
            this.label41 = new System.Windows.Forms.Label();
            this.nxTextBox = new System.Windows.Forms.TextBox();
            this.label42 = new System.Windows.Forms.Label();
            this.y1TextBox = new System.Windows.Forms.TextBox();
            this.label39 = new System.Windows.Forms.Label();
            this.x1TextBox = new System.Windows.Forms.TextBox();
            this.label40 = new System.Windows.Forms.Label();
            this.y0TextBox = new System.Windows.Forms.TextBox();
            this.label37 = new System.Windows.Forms.Label();
            this.x0TextBox = new System.Windows.Forms.TextBox();
            this.label38 = new System.Windows.Forms.Label();
            this.progressBar1 = new System.Windows.Forms.ProgressBar();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.folderBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
            this.timer2 = new System.Windows.Forms.Timer(this.components);
            this.tabControl1.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.tabPage3.SuspendLayout();
            this.tabPage2.SuspendLayout();
            this.tabPage4.SuspendLayout();
            this.tabPage5.SuspendLayout();
            this.tabPage6.SuspendLayout();
            this.flowLayoutPanel1.SuspendLayout();
            this.tabPage7.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.chart1)).BeginInit();
            this.tabPage8.SuspendLayout();
            this.tabPage9.SuspendLayout();
            this.tabPage10.SuspendLayout();
            this.tabPage11.SuspendLayout();
            this.SuspendLayout();
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.tabPage1);
            this.tabControl1.Controls.Add(this.tabPage3);
            this.tabControl1.Controls.Add(this.tabPage2);
            this.tabControl1.Controls.Add(this.tabPage4);
            this.tabControl1.Controls.Add(this.tabPage5);
            this.tabControl1.Controls.Add(this.tabPage6);
            this.tabControl1.Controls.Add(this.tabPage7);
            this.tabControl1.Controls.Add(this.tabPage8);
            this.tabControl1.Controls.Add(this.tabPage9);
            this.tabControl1.Controls.Add(this.tabPage10);
            this.tabControl1.Controls.Add(this.tabPage11);
            this.tabControl1.Font = new System.Drawing.Font("Microsoft Sans Serif", 11.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.tabControl1.Location = new System.Drawing.Point(3, 2);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(1212, 376);
            this.tabControl1.TabIndex = 0;
            this.tabControl1.SelectedIndexChanged += new System.EventHandler(this.tabControl1_SelectedIndexChanged);
            // 
            // tabPage1
            // 
            this.tabPage1.Controls.Add(this.oscilloscopeSwitch);
            this.tabPage1.Controls.Add(this.label26);
            this.tabPage1.Controls.Add(this.label1);
            this.tabPage1.Controls.Add(this.comboRangeA);
            this.tabPage1.Controls.Add(this.timebaseTextBox);
            this.tabPage1.Controls.Add(this.label13);
            this.tabPage1.Controls.Add(this.connectButton);
            this.tabPage1.Location = new System.Drawing.Point(4, 27);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage1.Size = new System.Drawing.Size(1204, 345);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "Подключение";
            this.tabPage1.UseVisualStyleBackColor = true;
            // 
            // oscilloscopeSwitch
            // 
            this.oscilloscopeSwitch.FormattingEnabled = true;
            this.oscilloscopeSwitch.Location = new System.Drawing.Point(17, 65);
            this.oscilloscopeSwitch.Name = "oscilloscopeSwitch";
            this.oscilloscopeSwitch.Size = new System.Drawing.Size(163, 26);
            this.oscilloscopeSwitch.TabIndex = 28;
            // 
            // label26
            // 
            this.label26.AutoSize = true;
            this.label26.Location = new System.Drawing.Point(265, 60);
            this.label26.Name = "label26";
            this.label26.Size = new System.Drawing.Size(166, 18);
            this.label26.TabIndex = 27;
            this.label26.Text = "Рекомендуется 100мВ";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.label1.Location = new System.Drawing.Point(294, 28);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(72, 16);
            this.label1.TabIndex = 26;
            this.label1.Text = "Диапазон";
            // 
            // comboRangeA
            // 
            this.comboRangeA.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.comboRangeA.FormattingEnabled = true;
            this.comboRangeA.Location = new System.Drawing.Point(434, 25);
            this.comboRangeA.Name = "comboRangeA";
            this.comboRangeA.Size = new System.Drawing.Size(121, 24);
            this.comboRangeA.TabIndex = 25;
            // 
            // timebaseTextBox
            // 
            this.timebaseTextBox.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.timebaseTextBox.Location = new System.Drawing.Point(126, 121);
            this.timebaseTextBox.Name = "timebaseTextBox";
            this.timebaseTextBox.Size = new System.Drawing.Size(100, 22);
            this.timebaseTextBox.TabIndex = 16;
            this.timebaseTextBox.Text = "4";
            this.timebaseTextBox.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.timebaseTextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.label13.Location = new System.Drawing.Point(25, 124);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(63, 16);
            this.label13.TabIndex = 15;
            this.label13.Text = "timebase";
            // 
            // connectButton
            // 
            this.connectButton.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.connectButton.Location = new System.Drawing.Point(17, 18);
            this.connectButton.Name = "connectButton";
            this.connectButton.Size = new System.Drawing.Size(164, 40);
            this.connectButton.TabIndex = 0;
            this.connectButton.Text = "Open Unit";
            this.connectButton.UseVisualStyleBackColor = true;
            this.connectButton.Click += new System.EventHandler(this.ConnectButton_Click);
            // 
            // tabPage3
            // 
            this.tabPage3.Controls.Add(this.disconnectAllButton);
            this.tabPage3.Controls.Add(this.label30);
            this.tabPage3.Controls.Add(this.layersTextBox);
            this.tabPage3.Controls.Add(this.invertSensorsButton);
            this.tabPage3.Controls.Add(this.sourcesCheckedListBox);
            this.tabPage3.Controls.Add(this.receiversCheckedListBox);
            this.tabPage3.Controls.Add(this.textBoxUnitInfo);
            this.tabPage3.Controls.Add(this.portListBox);
            this.tabPage3.Controls.Add(this.connectPortButton);
            this.tabPage3.Controls.Add(this.getPortsButton);
            this.tabPage3.Controls.Add(this.label16);
            this.tabPage3.Controls.Add(this.sensorTextBox);
            this.tabPage3.Controls.Add(this.label12);
            this.tabPage3.Controls.Add(this.label11);
            this.tabPage3.Controls.Add(this.label10);
            this.tabPage3.Controls.Add(this.disconnectButton);
            this.tabPage3.Controls.Add(this.label9);
            this.tabPage3.Controls.Add(this.label8);
            this.tabPage3.Controls.Add(this.label7);
            this.tabPage3.Controls.Add(this.label6);
            this.tabPage3.Controls.Add(this.label5);
            this.tabPage3.Controls.Add(this.label4);
            this.tabPage3.Controls.Add(this.label3);
            this.tabPage3.Controls.Add(this.label2);
            this.tabPage3.Controls.Add(this.connectAsSourceButton);
            this.tabPage3.Controls.Add(this.connectAsReceiverButton);
            this.tabPage3.Location = new System.Drawing.Point(4, 27);
            this.tabPage3.Name = "tabPage3";
            this.tabPage3.Size = new System.Drawing.Size(1204, 345);
            this.tabPage3.TabIndex = 2;
            this.tabPage3.Text = "Коммутация";
            this.tabPage3.UseVisualStyleBackColor = true;
            // 
            // disconnectAllButton
            // 
            this.disconnectAllButton.Location = new System.Drawing.Point(13, 287);
            this.disconnectAllButton.Name = "disconnectAllButton";
            this.disconnectAllButton.Size = new System.Drawing.Size(212, 36);
            this.disconnectAllButton.TabIndex = 42;
            this.disconnectAllButton.Text = "Отключить все";
            this.disconnectAllButton.UseVisualStyleBackColor = true;
            this.disconnectAllButton.Click += new System.EventHandler(this.DisconnectAllButton_Click);
            // 
            // label30
            // 
            this.label30.AutoSize = true;
            this.label30.Location = new System.Drawing.Point(10, 86);
            this.label30.Name = "label30";
            this.label30.Size = new System.Drawing.Size(62, 18);
            this.label30.TabIndex = 40;
            this.label30.Text = "СЛОЕВ";
            // 
            // layersTextBox
            // 
            this.layersTextBox.Location = new System.Drawing.Point(79, 80);
            this.layersTextBox.Name = "layersTextBox";
            this.layersTextBox.Size = new System.Drawing.Size(146, 24);
            this.layersTextBox.TabIndex = 41;
            this.layersTextBox.Text = "18";
            // 
            // invertSensorsButton
            // 
            this.invertSensorsButton.Location = new System.Drawing.Point(433, 293);
            this.invertSensorsButton.Name = "invertSensorsButton";
            this.invertSensorsButton.Size = new System.Drawing.Size(250, 32);
            this.invertSensorsButton.TabIndex = 39;
            this.invertSensorsButton.Text = "Инвертировать выбор";
            this.invertSensorsButton.UseVisualStyleBackColor = true;
            this.invertSensorsButton.Click += new System.EventHandler(this.InvertSensorsButton_Click);
            // 
            // sourcesCheckedListBox
            // 
            this.sourcesCheckedListBox.FormattingEnabled = true;
            this.sourcesCheckedListBox.Items.AddRange(new object[] {
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
            "32",
            "33",
            "34",
            "35",
            "36",
            "37",
            "38",
            "39",
            "40",
            "41",
            "42",
            "43",
            "44",
            "45",
            "46",
            "47",
            "48",
            "49",
            "50",
            "51",
            "52",
            "53",
            "54",
            "55"});
            this.sourcesCheckedListBox.Location = new System.Drawing.Point(613, 68);
            this.sourcesCheckedListBox.Name = "sourcesCheckedListBox";
            this.sourcesCheckedListBox.Size = new System.Drawing.Size(70, 156);
            this.sourcesCheckedListBox.TabIndex = 38;
            // 
            // receiversCheckedListBox
            // 
            this.receiversCheckedListBox.FormattingEnabled = true;
            this.receiversCheckedListBox.Items.AddRange(new object[] {
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
            "32",
            "33",
            "34",
            "35",
            "36",
            "37",
            "38",
            "39",
            "40",
            "41",
            "42",
            "43",
            "44",
            "45",
            "46",
            "47",
            "48",
            "49",
            "50",
            "51",
            "52",
            "53",
            "54",
            "55"});
            this.receiversCheckedListBox.Location = new System.Drawing.Point(520, 68);
            this.receiversCheckedListBox.Name = "receiversCheckedListBox";
            this.receiversCheckedListBox.Size = new System.Drawing.Size(64, 156);
            this.receiversCheckedListBox.TabIndex = 37;
            // 
            // textBoxUnitInfo
            // 
            this.textBoxUnitInfo.Location = new System.Drawing.Point(231, 18);
            this.textBoxUnitInfo.Multiline = true;
            this.textBoxUnitInfo.Name = "textBoxUnitInfo";
            this.textBoxUnitInfo.Size = new System.Drawing.Size(184, 307);
            this.textBoxUnitInfo.TabIndex = 36;
            // 
            // portListBox
            // 
            this.portListBox.FormattingEnabled = true;
            this.portListBox.ItemHeight = 18;
            this.portListBox.Location = new System.Drawing.Point(143, 16);
            this.portListBox.Name = "portListBox";
            this.portListBox.Size = new System.Drawing.Size(82, 58);
            this.portListBox.TabIndex = 35;
            // 
            // connectPortButton
            // 
            this.connectPortButton.Location = new System.Drawing.Point(13, 46);
            this.connectPortButton.Name = "connectPortButton";
            this.connectPortButton.Size = new System.Drawing.Size(124, 28);
            this.connectPortButton.TabIndex = 34;
            this.connectPortButton.Text = "Подключить порт";
            this.connectPortButton.UseVisualStyleBackColor = true;
            this.connectPortButton.Click += new System.EventHandler(this.ConnectPortButton_Click);
            // 
            // getPortsButton
            // 
            this.getPortsButton.Location = new System.Drawing.Point(13, 16);
            this.getPortsButton.Name = "getPortsButton";
            this.getPortsButton.Size = new System.Drawing.Size(124, 29);
            this.getPortsButton.TabIndex = 33;
            this.getPortsButton.Text = "Список портов";
            this.getPortsButton.UseVisualStyleBackColor = true;
            this.getPortsButton.Click += new System.EventHandler(this.GetPortsButton_Click);
            // 
            // label16
            // 
            this.label16.AutoSize = true;
            this.label16.Location = new System.Drawing.Point(436, 29);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(247, 18);
            this.label16.TabIndex = 32;
            this.label16.Text = "Роли датчиков в автоматическом";
            // 
            // sensorTextBox
            // 
            this.sensorTextBox.Location = new System.Drawing.Point(14, 128);
            this.sensorTextBox.Name = "sensorTextBox";
            this.sensorTextBox.Size = new System.Drawing.Size(211, 24);
            this.sensorTextBox.TabIndex = 31;
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Location = new System.Drawing.Point(11, 107);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(174, 18);
            this.label12.TabIndex = 30;
            this.label12.Text = "Введите номер датчика";
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(600, 47);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(83, 18);
            this.label11.TabIndex = 29;
            this.label11.Text = "Источники";
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(498, 49);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(86, 18);
            this.label10.TabIndex = 28;
            this.label10.Text = "Приемники";
            // 
            // disconnectButton
            // 
            this.disconnectButton.Location = new System.Drawing.Point(14, 238);
            this.disconnectButton.Name = "disconnectButton";
            this.disconnectButton.Size = new System.Drawing.Size(211, 36);
            this.disconnectButton.TabIndex = 27;
            this.disconnectButton.Text = "Отключить";
            this.disconnectButton.UseVisualStyleBackColor = true;
            this.disconnectButton.Click += new System.EventHandler(this.DisconnectButton_Click);
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(430, 206);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(86, 18);
            this.label9.TabIndex = 26;
            this.label9.Text = "Датчик H\\7";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(430, 184);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(87, 18);
            this.label8.TabIndex = 25;
            this.label8.Text = "Датчик G\\6";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(430, 164);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(84, 18);
            this.label7.TabIndex = 24;
            this.label7.Text = "Датчик F\\5";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(430, 146);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(85, 18);
            this.label6.TabIndex = 23;
            this.label6.Text = "Датчик E\\4";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(430, 128);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(86, 18);
            this.label5.TabIndex = 22;
            this.label5.Text = "Датчик D\\3";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(430, 109);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(86, 18);
            this.label4.TabIndex = 21;
            this.label4.Text = "Датчик C\\2";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(430, 91);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(85, 18);
            this.label3.TabIndex = 20;
            this.label3.Text = "Датчик B\\1";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(430, 70);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(84, 18);
            this.label2.TabIndex = 19;
            this.label2.Text = "Датчик A\\0";
            // 
            // connectAsSourceButton
            // 
            this.connectAsSourceButton.Location = new System.Drawing.Point(14, 198);
            this.connectAsSourceButton.Name = "connectAsSourceButton";
            this.connectAsSourceButton.Size = new System.Drawing.Size(211, 34);
            this.connectAsSourceButton.TabIndex = 18;
            this.connectAsSourceButton.Text = "Подключить как источник";
            this.connectAsSourceButton.UseVisualStyleBackColor = true;
            this.connectAsSourceButton.Click += new System.EventHandler(this.ConnectAsSourceButton_Click);
            // 
            // connectAsReceiverButton
            // 
            this.connectAsReceiverButton.Location = new System.Drawing.Point(14, 158);
            this.connectAsReceiverButton.Name = "connectAsReceiverButton";
            this.connectAsReceiverButton.Size = new System.Drawing.Size(211, 34);
            this.connectAsReceiverButton.TabIndex = 17;
            this.connectAsReceiverButton.Text = "Подключить как приемник";
            this.connectAsReceiverButton.UseVisualStyleBackColor = true;
            this.connectAsReceiverButton.Click += new System.EventHandler(this.ConnectAsReceiverButton_Click);
            // 
            // tabPage2
            // 
            this.tabPage2.Controls.Add(this.selectFolderButton);
            this.tabPage2.Controls.Add(this.label28);
            this.tabPage2.Controls.Add(this.fileNameTextBox);
            this.tabPage2.Controls.Add(this.saveEveryTextBox);
            this.tabPage2.Controls.Add(this.label27);
            this.tabPage2.Controls.Add(this.saveRawCheckBox);
            this.tabPage2.Controls.Add(this.saveTimesCheckBox);
            this.tabPage2.Controls.Add(this.FTStepsNumberTextBox);
            this.tabPage2.Controls.Add(this.FTStepTextBox);
            this.tabPage2.Controls.Add(this.FTBottomTextBox);
            this.tabPage2.Controls.Add(this.label22);
            this.tabPage2.Controls.Add(this.label19);
            this.tabPage2.Controls.Add(this.label18);
            this.tabPage2.Controls.Add(this.computeModuleCheckBox);
            this.tabPage2.Controls.Add(this.computeFTCheckBox);
            this.tabPage2.Controls.Add(this.label17);
            this.tabPage2.Controls.Add(this.filePathTextBox);
            this.tabPage2.Controls.Add(this.suppressStartCheckBox);
            this.tabPage2.Controls.Add(this.autocollectButton);
            this.tabPage2.Controls.Add(this.visualizationCheckBox);
            this.tabPage2.Controls.Add(this.stopButton);
            this.tabPage2.Controls.Add(this.passBandLimit20CheckBox);
            this.tabPage2.Controls.Add(this.suppressedTextBox);
            this.tabPage2.Controls.Add(this.label21);
            this.tabPage2.Controls.Add(this.stepsBeforeTextBox);
            this.tabPage2.Controls.Add(this.label20);
            this.tabPage2.Controls.Add(this.label15);
            this.tabPage2.Controls.Add(this.averagingsTextBox);
            this.tabPage2.Controls.Add(this.stepsAfterTextBox);
            this.tabPage2.Controls.Add(this.label14);
            this.tabPage2.Controls.Add(this.button1);
            this.tabPage2.Controls.Add(this.collectDataButton);
            this.tabPage2.Location = new System.Drawing.Point(4, 27);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage2.Size = new System.Drawing.Size(1204, 345);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "Сбор данных";
            this.tabPage2.UseVisualStyleBackColor = true;
            // 
            // selectFolderButton
            // 
            this.selectFolderButton.Location = new System.Drawing.Point(366, 254);
            this.selectFolderButton.Name = "selectFolderButton";
            this.selectFolderButton.Size = new System.Drawing.Size(198, 30);
            this.selectFolderButton.TabIndex = 54;
            this.selectFolderButton.Text = "Выбрать папку";
            this.selectFolderButton.UseVisualStyleBackColor = true;
            this.selectFolderButton.Click += new System.EventHandler(this.SelectFolderButton_Click);
            // 
            // label28
            // 
            this.label28.AutoSize = true;
            this.label28.Location = new System.Drawing.Point(3, 291);
            this.label28.Name = "label28";
            this.label28.Size = new System.Drawing.Size(38, 18);
            this.label28.TabIndex = 53;
            this.label28.Text = "Имя";
            // 
            // fileNameTextBox
            // 
            this.fileNameTextBox.Location = new System.Drawing.Point(117, 285);
            this.fileNameTextBox.Name = "fileNameTextBox";
            this.fileNameTextBox.Size = new System.Drawing.Size(571, 24);
            this.fileNameTextBox.TabIndex = 52;
            this.fileNameTextBox.Text = "test_com2.txt";
            // 
            // saveEveryTextBox
            // 
            this.saveEveryTextBox.Location = new System.Drawing.Point(542, 204);
            this.saveEveryTextBox.Name = "saveEveryTextBox";
            this.saveEveryTextBox.Size = new System.Drawing.Size(100, 24);
            this.saveEveryTextBox.TabIndex = 51;
            this.saveEveryTextBox.Text = "1";
            this.saveEveryTextBox.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.saveEveryTextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // label27
            // 
            this.label27.AutoSize = true;
            this.label27.Location = new System.Drawing.Point(395, 206);
            this.label27.Name = "label27";
            this.label27.Size = new System.Drawing.Size(141, 18);
            this.label27.TabIndex = 50;
            this.label27.Text = "Сохранять каждый";
            // 
            // saveRawCheckBox
            // 
            this.saveRawCheckBox.AutoSize = true;
            this.saveRawCheckBox.Checked = true;
            this.saveRawCheckBox.CheckState = System.Windows.Forms.CheckState.Checked;
            this.saveRawCheckBox.Location = new System.Drawing.Point(452, 77);
            this.saveRawCheckBox.Name = "saveRawCheckBox";
            this.saveRawCheckBox.Size = new System.Drawing.Size(196, 22);
            this.saveRawCheckBox.TabIndex = 49;
            this.saveRawCheckBox.Text = "Сохранять сырой замер";
            this.saveRawCheckBox.UseVisualStyleBackColor = true;
            // 
            // saveTimesCheckBox
            // 
            this.saveTimesCheckBox.AutoSize = true;
            this.saveTimesCheckBox.Checked = true;
            this.saveTimesCheckBox.CheckState = System.Windows.Forms.CheckState.Checked;
            this.saveTimesCheckBox.Location = new System.Drawing.Point(451, 48);
            this.saveTimesCheckBox.Name = "saveTimesCheckBox";
            this.saveTimesCheckBox.Size = new System.Drawing.Size(237, 22);
            this.saveTimesCheckBox.TabIndex = 48;
            this.saveTimesCheckBox.Text = "Сохранять файл с временами";
            this.saveTimesCheckBox.UseVisualStyleBackColor = true;
            // 
            // FTStepsNumberTextBox
            // 
            this.FTStepsNumberTextBox.Location = new System.Drawing.Point(260, 251);
            this.FTStepsNumberTextBox.Name = "FTStepsNumberTextBox";
            this.FTStepsNumberTextBox.Size = new System.Drawing.Size(100, 24);
            this.FTStepsNumberTextBox.TabIndex = 47;
            this.FTStepsNumberTextBox.Text = "7000";
            this.FTStepsNumberTextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // FTStepTextBox
            // 
            this.FTStepTextBox.Location = new System.Drawing.Point(260, 224);
            this.FTStepTextBox.Name = "FTStepTextBox";
            this.FTStepTextBox.Size = new System.Drawing.Size(100, 24);
            this.FTStepTextBox.TabIndex = 46;
            this.FTStepTextBox.Text = "100";
            this.FTStepTextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // FTBottomTextBox
            // 
            this.FTBottomTextBox.Location = new System.Drawing.Point(260, 196);
            this.FTBottomTextBox.Name = "FTBottomTextBox";
            this.FTBottomTextBox.Size = new System.Drawing.Size(100, 24);
            this.FTBottomTextBox.TabIndex = 45;
            this.FTBottomTextBox.Text = "100";
            this.FTBottomTextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // label22
            // 
            this.label22.AutoSize = true;
            this.label22.Location = new System.Drawing.Point(8, 254);
            this.label22.Name = "label22";
            this.label22.Size = new System.Drawing.Size(246, 18);
            this.label22.TabIndex = 44;
            this.label22.Text = "Количество шагов по частоте ПФ";
            // 
            // label19
            // 
            this.label19.AutoSize = true;
            this.label19.Location = new System.Drawing.Point(8, 230);
            this.label19.Name = "label19";
            this.label19.Size = new System.Drawing.Size(143, 18);
            this.label19.TabIndex = 43;
            this.label19.Text = "Шаг по частоте ПФ";
            // 
            // label18
            // 
            this.label18.AutoSize = true;
            this.label18.Location = new System.Drawing.Point(8, 204);
            this.label18.Name = "label18";
            this.label18.Size = new System.Drawing.Size(149, 18);
            this.label18.TabIndex = 42;
            this.label18.Text = "Нижняя частота ПФ";
            // 
            // computeModuleCheckBox
            // 
            this.computeModuleCheckBox.AutoSize = true;
            this.computeModuleCheckBox.Location = new System.Drawing.Point(11, 179);
            this.computeModuleCheckBox.Name = "computeModuleCheckBox";
            this.computeModuleCheckBox.Size = new System.Drawing.Size(320, 22);
            this.computeModuleCheckBox.TabIndex = 41;
            this.computeModuleCheckBox.Text = "Если считается ПФ, то посчитать модуль";
            this.computeModuleCheckBox.UseVisualStyleBackColor = true;
            // 
            // computeFTCheckBox
            // 
            this.computeFTCheckBox.AutoSize = true;
            this.computeFTCheckBox.Location = new System.Drawing.Point(11, 155);
            this.computeFTCheckBox.Name = "computeFTCheckBox";
            this.computeFTCheckBox.Size = new System.Drawing.Size(112, 22);
            this.computeFTCheckBox.TabIndex = 40;
            this.computeFTCheckBox.Text = "Считать ПФ";
            this.computeFTCheckBox.UseVisualStyleBackColor = true;
            // 
            // label17
            // 
            this.label17.AutoSize = true;
            this.label17.Location = new System.Drawing.Point(3, 321);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(108, 18);
            this.label17.TabIndex = 39;
            this.label17.Text = "Путь хранения";
            // 
            // filePathTextBox
            // 
            this.filePathTextBox.Location = new System.Drawing.Point(117, 315);
            this.filePathTextBox.Name = "filePathTextBox";
            this.filePathTextBox.Size = new System.Drawing.Size(571, 24);
            this.filePathTextBox.TabIndex = 38;
            this.filePathTextBox.Text = "C:\\testhead10\\";
            // 
            // suppressStartCheckBox
            // 
            this.suppressStartCheckBox.AutoSize = true;
            this.suppressStartCheckBox.Location = new System.Drawing.Point(451, 20);
            this.suppressStartCheckBox.Name = "suppressStartCheckBox";
            this.suppressStartCheckBox.Size = new System.Drawing.Size(226, 22);
            this.suppressStartCheckBox.TabIndex = 37;
            this.suppressStartCheckBox.Text = "Подавлять начальную часть";
            this.suppressStartCheckBox.UseVisualStyleBackColor = true;
            // 
            // autocollectButton
            // 
            this.autocollectButton.Location = new System.Drawing.Point(7, 6);
            this.autocollectButton.Name = "autocollectButton";
            this.autocollectButton.Size = new System.Drawing.Size(182, 35);
            this.autocollectButton.TabIndex = 36;
            this.autocollectButton.Text = "Автоматический сбор данных";
            this.autocollectButton.UseVisualStyleBackColor = true;
            this.autocollectButton.Click += new System.EventHandler(this.button11_Click);
            // 
            // visualizationCheckBox
            // 
            this.visualizationCheckBox.AutoSize = true;
            this.visualizationCheckBox.Location = new System.Drawing.Point(11, 131);
            this.visualizationCheckBox.Name = "visualizationCheckBox";
            this.visualizationCheckBox.Size = new System.Drawing.Size(125, 22);
            this.visualizationCheckBox.TabIndex = 35;
            this.visualizationCheckBox.Text = "Визуализация";
            this.visualizationCheckBox.UseVisualStyleBackColor = true;
            // 
            // stopButton
            // 
            this.stopButton.Location = new System.Drawing.Point(7, 48);
            this.stopButton.Name = "stopButton";
            this.stopButton.Size = new System.Drawing.Size(182, 32);
            this.stopButton.TabIndex = 34;
            this.stopButton.Text = "Стоп";
            this.stopButton.UseVisualStyleBackColor = true;
            this.stopButton.Click += new System.EventHandler(this.StopButton_Click);
            // 
            // passBandLimit20CheckBox
            // 
            this.passBandLimit20CheckBox.AutoSize = true;
            this.passBandLimit20CheckBox.Checked = true;
            this.passBandLimit20CheckBox.CheckState = System.Windows.Forms.CheckState.Checked;
            this.passBandLimit20CheckBox.Location = new System.Drawing.Point(228, 154);
            this.passBandLimit20CheckBox.Name = "passBandLimit20CheckBox";
            this.passBandLimit20CheckBox.Size = new System.Drawing.Size(321, 22);
            this.passBandLimit20CheckBox.TabIndex = 33;
            this.passBandLimit20CheckBox.Text = "Ограничение полосы пропускания 20 МГц";
            this.passBandLimit20CheckBox.UseVisualStyleBackColor = true;
            // 
            // suppressedTextBox
            // 
            this.suppressedTextBox.Location = new System.Drawing.Point(345, 21);
            this.suppressedTextBox.Name = "suppressedTextBox";
            this.suppressedTextBox.Size = new System.Drawing.Size(100, 24);
            this.suppressedTextBox.TabIndex = 32;
            this.suppressedTextBox.Text = "6300";
            this.suppressedTextBox.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.suppressedTextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // label21
            // 
            this.label21.AutoSize = true;
            this.label21.Location = new System.Drawing.Point(195, 24);
            this.label21.Name = "label21";
            this.label21.Size = new System.Drawing.Size(156, 18);
            this.label21.TabIndex = 31;
            this.label21.Text = "Число подавляемых ";
            // 
            // stepsBeforeTextBox
            // 
            this.stepsBeforeTextBox.Location = new System.Drawing.Point(345, 56);
            this.stepsBeforeTextBox.Name = "stepsBeforeTextBox";
            this.stepsBeforeTextBox.Size = new System.Drawing.Size(100, 24);
            this.stepsBeforeTextBox.TabIndex = 30;
            this.stepsBeforeTextBox.Text = "5000";
            this.stepsBeforeTextBox.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.stepsBeforeTextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // label20
            // 
            this.label20.AutoSize = true;
            this.label20.Location = new System.Drawing.Point(195, 59);
            this.label20.Name = "label20";
            this.label20.Size = new System.Drawing.Size(120, 18);
            this.label20.TabIndex = 29;
            this.label20.Text = "Число шагов до";
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.Location = new System.Drawing.Point(195, 132);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(136, 18);
            this.label15.TabIndex = 28;
            this.label15.Text = "Число усреднений";
            // 
            // averagingsTextBox
            // 
            this.averagingsTextBox.Location = new System.Drawing.Point(345, 128);
            this.averagingsTextBox.Name = "averagingsTextBox";
            this.averagingsTextBox.Size = new System.Drawing.Size(100, 24);
            this.averagingsTextBox.TabIndex = 27;
            this.averagingsTextBox.Text = "500";
            this.averagingsTextBox.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.averagingsTextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // stepsAfterTextBox
            // 
            this.stepsAfterTextBox.Location = new System.Drawing.Point(345, 93);
            this.stepsAfterTextBox.Name = "stepsAfterTextBox";
            this.stepsAfterTextBox.Size = new System.Drawing.Size(100, 24);
            this.stepsAfterTextBox.TabIndex = 26;
            this.stepsAfterTextBox.Text = "95000";
            this.stepsAfterTextBox.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.stepsAfterTextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Location = new System.Drawing.Point(195, 99);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(144, 18);
            this.label14.TabIndex = 25;
            this.label14.Text = "Число шагов после";
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(694, 254);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(103, 30);
            this.button1.TabIndex = 0;
            this.button1.Text = "Сбор2";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // collectDataButton
            // 
            this.collectDataButton.Location = new System.Drawing.Point(585, 254);
            this.collectDataButton.Name = "collectDataButton";
            this.collectDataButton.Size = new System.Drawing.Size(103, 30);
            this.collectDataButton.TabIndex = 0;
            this.collectDataButton.Text = "Сбор";
            this.collectDataButton.UseVisualStyleBackColor = true;
            this.collectDataButton.Click += new System.EventHandler(this.CollectDataButton_Click);
            // 
            // tabPage4
            // 
            this.tabPage4.Controls.Add(this.useFreqFilterTextBox);
            this.tabPage4.Controls.Add(this.useFreqFilterCheckBox);
            this.tabPage4.Controls.Add(this.useTimeFilterTextBox);
            this.tabPage4.Controls.Add(this.useTimeFilterCheckBox);
            this.tabPage4.Controls.Add(this.autoEliminateAvgCheckBox);
            this.tabPage4.Controls.Add(this.applyAutoRunAvgCheckBox);
            this.tabPage4.Controls.Add(this.eliminateAvgButton);
            this.tabPage4.Controls.Add(this.runAvgTextBox);
            this.tabPage4.Controls.Add(this.applyRunAvgButton);
            this.tabPage4.Location = new System.Drawing.Point(4, 27);
            this.tabPage4.Name = "tabPage4";
            this.tabPage4.Size = new System.Drawing.Size(1204, 345);
            this.tabPage4.TabIndex = 3;
            this.tabPage4.Text = "Предобработка";
            this.tabPage4.UseVisualStyleBackColor = true;
            // 
            // useFreqFilterTextBox
            // 
            this.useFreqFilterTextBox.Location = new System.Drawing.Point(6, 185);
            this.useFreqFilterTextBox.Name = "useFreqFilterTextBox";
            this.useFreqFilterTextBox.Size = new System.Drawing.Size(646, 24);
            this.useFreqFilterTextBox.TabIndex = 8;
            this.useFreqFilterTextBox.Text = "C:\\testhead10\\my_filter_f200.txt";
            // 
            // useFreqFilterCheckBox
            // 
            this.useFreqFilterCheckBox.AutoSize = true;
            this.useFreqFilterCheckBox.Location = new System.Drawing.Point(4, 161);
            this.useFreqFilterCheckBox.Name = "useFreqFilterCheckBox";
            this.useFreqFilterCheckBox.Size = new System.Drawing.Size(335, 22);
            this.useFreqFilterCheckBox.TabIndex = 7;
            this.useFreqFilterCheckBox.Text = "Использовать фильтр в частотной области";
            this.useFreqFilterCheckBox.UseVisualStyleBackColor = true;
            // 
            // useTimeFilterTextBox
            // 
            this.useTimeFilterTextBox.Location = new System.Drawing.Point(5, 134);
            this.useTimeFilterTextBox.Name = "useTimeFilterTextBox";
            this.useTimeFilterTextBox.Size = new System.Drawing.Size(647, 24);
            this.useTimeFilterTextBox.TabIndex = 6;
            this.useTimeFilterTextBox.Text = "C:\\TEMP\\my_filter_t.txt";
            // 
            // useTimeFilterCheckBox
            // 
            this.useTimeFilterCheckBox.AutoSize = true;
            this.useTimeFilterCheckBox.Location = new System.Drawing.Point(6, 111);
            this.useTimeFilterCheckBox.Name = "useTimeFilterCheckBox";
            this.useTimeFilterCheckBox.Size = new System.Drawing.Size(339, 22);
            this.useTimeFilterCheckBox.TabIndex = 5;
            this.useTimeFilterCheckBox.Text = "Использовать фильтр в временной области";
            this.useTimeFilterCheckBox.UseVisualStyleBackColor = true;
            // 
            // autoEliminateAvgCheckBox
            // 
            this.autoEliminateAvgCheckBox.AutoSize = true;
            this.autoEliminateAvgCheckBox.Checked = true;
            this.autoEliminateAvgCheckBox.CheckState = System.Windows.Forms.CheckState.Checked;
            this.autoEliminateAvgCheckBox.Location = new System.Drawing.Point(143, 72);
            this.autoEliminateAvgCheckBox.Name = "autoEliminateAvgCheckBox";
            this.autoEliminateAvgCheckBox.Size = new System.Drawing.Size(570, 22);
            this.autoEliminateAvgCheckBox.TabIndex = 4;
            this.autoEliminateAvgCheckBox.Text = "Применять устранение средней величины при автоматическом сборе данных";
            this.autoEliminateAvgCheckBox.UseVisualStyleBackColor = true;
            // 
            // applyAutoRunAvgCheckBox
            // 
            this.applyAutoRunAvgCheckBox.AutoSize = true;
            this.applyAutoRunAvgCheckBox.Checked = true;
            this.applyAutoRunAvgCheckBox.CheckState = System.Windows.Forms.CheckState.Checked;
            this.applyAutoRunAvgCheckBox.Location = new System.Drawing.Point(143, 44);
            this.applyAutoRunAvgCheckBox.Name = "applyAutoRunAvgCheckBox";
            this.applyAutoRunAvgCheckBox.Size = new System.Drawing.Size(477, 22);
            this.applyAutoRunAvgCheckBox.TabIndex = 3;
            this.applyAutoRunAvgCheckBox.Text = "Применять бегущее среднее при автоматическом сборе данных";
            this.applyAutoRunAvgCheckBox.UseVisualStyleBackColor = true;
            // 
            // eliminateAvgButton
            // 
            this.eliminateAvgButton.Location = new System.Drawing.Point(5, 57);
            this.eliminateAvgButton.Name = "eliminateAvgButton";
            this.eliminateAvgButton.Size = new System.Drawing.Size(132, 37);
            this.eliminateAvgButton.TabIndex = 2;
            this.eliminateAvgButton.Text = "Устранить среднюю величину";
            this.eliminateAvgButton.UseVisualStyleBackColor = true;
            this.eliminateAvgButton.Click += new System.EventHandler(this.EliminateAvgButton_Click);
            // 
            // runAvgTextBox
            // 
            this.runAvgTextBox.Location = new System.Drawing.Point(143, 14);
            this.runAvgTextBox.Name = "runAvgTextBox";
            this.runAvgTextBox.Size = new System.Drawing.Size(100, 24);
            this.runAvgTextBox.TabIndex = 1;
            this.runAvgTextBox.Text = "5";
            // 
            // applyRunAvgButton
            // 
            this.applyRunAvgButton.Location = new System.Drawing.Point(5, 14);
            this.applyRunAvgButton.Name = "applyRunAvgButton";
            this.applyRunAvgButton.Size = new System.Drawing.Size(132, 37);
            this.applyRunAvgButton.TabIndex = 0;
            this.applyRunAvgButton.Text = "Применить бегущее среднее";
            this.applyRunAvgButton.UseVisualStyleBackColor = true;
            this.applyRunAvgButton.Click += new System.EventHandler(this.ApplyRunAvgButton_Click);
            // 
            // tabPage5
            // 
            this.tabPage5.Controls.Add(this.applyFreqFilterCheckBox);
            this.tabPage5.Controls.Add(this.suppressToTextBox);
            this.tabPage5.Controls.Add(this.suppressToCheckBox);
            this.tabPage5.Controls.Add(this.differenceInfoTextBox);
            this.tabPage5.Controls.Add(this.diffFolderTextBox);
            this.tabPage5.Controls.Add(this.label25);
            this.tabPage5.Controls.Add(this.normBeforeOutputCheckBox);
            this.tabPage5.Controls.Add(this.buildDifferencesButton);
            this.tabPage5.Controls.Add(this.defectFolderTextBox);
            this.tabPage5.Controls.Add(this.label24);
            this.tabPage5.Controls.Add(this.label23);
            this.tabPage5.Controls.Add(this.noDefectFolderTextBox);
            this.tabPage5.Location = new System.Drawing.Point(4, 27);
            this.tabPage5.Name = "tabPage5";
            this.tabPage5.Size = new System.Drawing.Size(1204, 345);
            this.tabPage5.TabIndex = 4;
            this.tabPage5.Text = "Обработка";
            this.tabPage5.UseVisualStyleBackColor = true;
            // 
            // applyFreqFilterCheckBox
            // 
            this.applyFreqFilterCheckBox.AutoSize = true;
            this.applyFreqFilterCheckBox.Location = new System.Drawing.Point(376, 143);
            this.applyFreqFilterCheckBox.Name = "applyFreqFilterCheckBox";
            this.applyFreqFilterCheckBox.Size = new System.Drawing.Size(241, 22);
            this.applyFreqFilterCheckBox.TabIndex = 11;
            this.applyFreqFilterCheckBox.Text = "Применить фильтр по частоте";
            this.applyFreqFilterCheckBox.UseVisualStyleBackColor = true;
            // 
            // suppressToTextBox
            // 
            this.suppressToTextBox.Location = new System.Drawing.Point(531, 103);
            this.suppressToTextBox.Name = "suppressToTextBox";
            this.suppressToTextBox.Size = new System.Drawing.Size(151, 24);
            this.suppressToTextBox.TabIndex = 10;
            this.suppressToTextBox.Text = "5100";
            // 
            // suppressToCheckBox
            // 
            this.suppressToCheckBox.AutoSize = true;
            this.suppressToCheckBox.Location = new System.Drawing.Point(242, 103);
            this.suppressToCheckBox.Name = "suppressToCheckBox";
            this.suppressToCheckBox.Size = new System.Drawing.Size(126, 22);
            this.suppressToCheckBox.TabIndex = 9;
            this.suppressToCheckBox.Text = "Подавлять до";
            this.suppressToCheckBox.UseVisualStyleBackColor = true;
            // 
            // differenceInfoTextBox
            // 
            this.differenceInfoTextBox.Location = new System.Drawing.Point(8, 172);
            this.differenceInfoTextBox.Multiline = true;
            this.differenceInfoTextBox.Name = "differenceInfoTextBox";
            this.differenceInfoTextBox.Size = new System.Drawing.Size(674, 170);
            this.differenceInfoTextBox.TabIndex = 8;
            this.differenceInfoTextBox.Text = "Информация о построенном разностном замере";
            // 
            // diffFolderTextBox
            // 
            this.diffFolderTextBox.Location = new System.Drawing.Point(242, 66);
            this.diffFolderTextBox.Name = "diffFolderTextBox";
            this.diffFolderTextBox.Size = new System.Drawing.Size(440, 24);
            this.diffFolderTextBox.TabIndex = 7;
            this.diffFolderTextBox.Text = "C:\\Users\\user1\\Desktop\\ext_new\\16mm\\razn\\";
            // 
            // label25
            // 
            this.label25.AutoSize = true;
            this.label25.Location = new System.Drawing.Point(5, 72);
            this.label25.Name = "label25";
            this.label25.Size = new System.Drawing.Size(241, 18);
            this.label25.TabIndex = 6;
            this.label25.Text = "Папка для сохранения разностей";
            // 
            // normBeforeOutputCheckBox
            // 
            this.normBeforeOutputCheckBox.AutoSize = true;
            this.normBeforeOutputCheckBox.Location = new System.Drawing.Point(8, 143);
            this.normBeforeOutputCheckBox.Name = "normBeforeOutputCheckBox";
            this.normBeforeOutputCheckBox.Size = new System.Drawing.Size(326, 22);
            this.normBeforeOutputCheckBox.TabIndex = 5;
            this.normBeforeOutputCheckBox.Text = "Нормализовывать перед выводом в файл";
            this.normBeforeOutputCheckBox.UseVisualStyleBackColor = true;
            // 
            // buildDifferencesButton
            // 
            this.buildDifferencesButton.Location = new System.Drawing.Point(8, 99);
            this.buildDifferencesButton.Name = "buildDifferencesButton";
            this.buildDifferencesButton.Size = new System.Drawing.Size(167, 28);
            this.buildDifferencesButton.TabIndex = 4;
            this.buildDifferencesButton.Text = "Построить разности";
            this.buildDifferencesButton.UseVisualStyleBackColor = true;
            this.buildDifferencesButton.Click += new System.EventHandler(this.BuildDifferencesButton_Click);
            // 
            // defectFolderTextBox
            // 
            this.defectFolderTextBox.Location = new System.Drawing.Point(242, 40);
            this.defectFolderTextBox.Name = "defectFolderTextBox";
            this.defectFolderTextBox.Size = new System.Drawing.Size(440, 24);
            this.defectFolderTextBox.TabIndex = 3;
            this.defectFolderTextBox.Text = "C:\\Users\\user1\\Desktop\\ext_new\\16mm\\def\\";
            // 
            // label24
            // 
            this.label24.AutoSize = true;
            this.label24.Location = new System.Drawing.Point(5, 46);
            this.label24.Name = "label24";
            this.label24.Size = new System.Drawing.Size(226, 18);
            this.label24.TabIndex = 2;
            this.label24.Text = "Папка с замерами с дефектом";
            // 
            // label23
            // 
            this.label23.AutoSize = true;
            this.label23.Location = new System.Drawing.Point(5, 21);
            this.label23.Name = "label23";
            this.label23.Size = new System.Drawing.Size(231, 18);
            this.label23.TabIndex = 1;
            this.label23.Text = "Папка с замерами без дефекта";
            // 
            // noDefectFolderTextBox
            // 
            this.noDefectFolderTextBox.Location = new System.Drawing.Point(242, 15);
            this.noDefectFolderTextBox.Name = "noDefectFolderTextBox";
            this.noDefectFolderTextBox.Size = new System.Drawing.Size(440, 24);
            this.noDefectFolderTextBox.TabIndex = 0;
            this.noDefectFolderTextBox.Text = "C:\\Users\\user1\\Desktop\\ext_new\\16mm\\bez\\";
            // 
            // tabPage6
            // 
            this.tabPage6.Controls.Add(this.flowLayoutPanel1);
            this.tabPage6.Location = new System.Drawing.Point(4, 27);
            this.tabPage6.Name = "tabPage6";
            this.tabPage6.Size = new System.Drawing.Size(1204, 345);
            this.tabPage6.TabIndex = 11;
            this.tabPage6.Text = "Фильтр";
            this.tabPage6.UseVisualStyleBackColor = true;
            // 
            // flowLayoutPanel1
            // 
            this.flowLayoutPanel1.Controls.Add(this.loadMeteringButton);
            this.flowLayoutPanel1.Controls.Add(this.saveMeteringButton);
            this.flowLayoutPanel1.Controls.Add(this.applyFilterButton);
            this.flowLayoutPanel1.Controls.Add(this.filtersComboBox);
            this.flowLayoutPanel1.Controls.Add(this.undoLastFilterButton);
            this.flowLayoutPanel1.Controls.Add(this.filtersHistoryTextBox);
            this.flowLayoutPanel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.flowLayoutPanel1.Location = new System.Drawing.Point(0, 0);
            this.flowLayoutPanel1.Name = "flowLayoutPanel1";
            this.flowLayoutPanel1.Size = new System.Drawing.Size(1204, 345);
            this.flowLayoutPanel1.TabIndex = 0;
            // 
            // loadMeteringButton
            // 
            this.loadMeteringButton.AutoSize = true;
            this.loadMeteringButton.Location = new System.Drawing.Point(3, 3);
            this.loadMeteringButton.Name = "loadMeteringButton";
            this.loadMeteringButton.Size = new System.Drawing.Size(252, 39);
            this.loadMeteringButton.TabIndex = 0;
            this.loadMeteringButton.Text = "Загрузить из файла";
            this.loadMeteringButton.UseVisualStyleBackColor = true;
            this.loadMeteringButton.Click += new System.EventHandler(this.loadMeteringButton_Click);
            // 
            // saveMeteringButton
            // 
            this.saveMeteringButton.AutoSize = true;
            this.flowLayoutPanel1.SetFlowBreak(this.saveMeteringButton, true);
            this.saveMeteringButton.Location = new System.Drawing.Point(261, 3);
            this.saveMeteringButton.Name = "saveMeteringButton";
            this.saveMeteringButton.Size = new System.Drawing.Size(230, 39);
            this.saveMeteringButton.TabIndex = 1;
            this.saveMeteringButton.Text = "Сохранить в файл";
            this.saveMeteringButton.UseVisualStyleBackColor = true;
            this.saveMeteringButton.Click += new System.EventHandler(this.saveMeteringButton_Click);
            // 
            // applyFilterButton
            // 
            this.applyFilterButton.AutoSize = true;
            this.applyFilterButton.Location = new System.Drawing.Point(3, 48);
            this.applyFilterButton.Name = "applyFilterButton";
            this.applyFilterButton.Size = new System.Drawing.Size(246, 39);
            this.applyFilterButton.TabIndex = 2;
            this.applyFilterButton.Text = "Применить фильтр";
            this.applyFilterButton.UseVisualStyleBackColor = true;
            this.applyFilterButton.Click += new System.EventHandler(this.applyFilterButton_Click);
            // 
            // filtersComboBox
            // 
            this.filtersComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.filtersComboBox.FormattingEnabled = true;
            this.filtersComboBox.Items.AddRange(new object[] {
            "Среднее арифметическое",
            "Медианный фильтр",
            "Экспоненциальное бегущее среднее",
            "Фурье-удаление низких частот",
            "Фурье-удаление высоких частот"});
            this.filtersComboBox.Location = new System.Drawing.Point(255, 48);
            this.filtersComboBox.Name = "filtersComboBox";
            this.filtersComboBox.Size = new System.Drawing.Size(121, 26);
            this.filtersComboBox.TabIndex = 3;
            // 
            // undoLastFilterButton
            // 
            this.undoLastFilterButton.AutoSize = true;
            this.flowLayoutPanel1.SetFlowBreak(this.undoLastFilterButton, true);
            this.undoLastFilterButton.Location = new System.Drawing.Point(382, 48);
            this.undoLastFilterButton.Name = "undoLastFilterButton";
            this.undoLastFilterButton.Size = new System.Drawing.Size(359, 39);
            this.undoLastFilterButton.TabIndex = 4;
            this.undoLastFilterButton.Text = "Отменить последний фильтр";
            this.undoLastFilterButton.UseVisualStyleBackColor = true;
            this.undoLastFilterButton.Click += new System.EventHandler(this.undoLastFilterButton_Click);
            // 
            // filtersHistoryTextBox
            // 
            this.filtersHistoryTextBox.Location = new System.Drawing.Point(3, 93);
            this.filtersHistoryTextBox.Multiline = true;
            this.filtersHistoryTextBox.Name = "filtersHistoryTextBox";
            this.filtersHistoryTextBox.ReadOnly = true;
            this.filtersHistoryTextBox.Size = new System.Drawing.Size(1023, 207);
            this.filtersHistoryTextBox.TabIndex = 5;
            // 
            // tabPage7
            // 
            this.tabPage7.Controls.Add(this.chart1);
            this.tabPage7.Location = new System.Drawing.Point(4, 27);
            this.tabPage7.Name = "tabPage7";
            this.tabPage7.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage7.Size = new System.Drawing.Size(1204, 345);
            this.tabPage7.TabIndex = 6;
            this.tabPage7.Text = "Визуализация";
            this.tabPage7.UseVisualStyleBackColor = true;
            // 
            // chart1
            // 
            chartArea1.Name = "ChartArea1";
            this.chart1.ChartAreas.Add(chartArea1);
            legend1.Name = "Legend1";
            this.chart1.Legends.Add(legend1);
            this.chart1.Location = new System.Drawing.Point(0, 3);
            this.chart1.Name = "chart1";
            series1.ChartArea = "ChartArea1";
            series1.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.FastLine;
            series1.Legend = "Legend1";
            series1.Name = "Series1";
            this.chart1.Series.Add(series1);
            this.chart1.Size = new System.Drawing.Size(688, 342);
            this.chart1.TabIndex = 2;
            this.chart1.Text = "chart1";
            // 
            // tabPage8
            // 
            this.tabPage8.Controls.Add(this.button15);
            this.tabPage8.Location = new System.Drawing.Point(4, 27);
            this.tabPage8.Name = "tabPage8";
            this.tabPage8.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage8.Size = new System.Drawing.Size(1204, 345);
            this.tabPage8.TabIndex = 7;
            this.tabPage8.Text = "TEstinGGGG";
            this.tabPage8.UseVisualStyleBackColor = true;
            // 
            // button15
            // 
            this.button15.Location = new System.Drawing.Point(18, 22);
            this.button15.Name = "button15";
            this.button15.Size = new System.Drawing.Size(75, 23);
            this.button15.TabIndex = 0;
            this.button15.Text = "button15";
            this.button15.UseVisualStyleBackColor = true;
            this.button15.Click += new System.EventHandler(this.button15_Click);
            // 
            // tabPage9
            // 
            this.tabPage9.Controls.Add(this.normalizeCheckBox);
            this.tabPage9.Controls.Add(this.computeFTButton);
            this.tabPage9.Controls.Add(this.label29);
            this.tabPage9.Controls.Add(this.measurementFolderTextBox);
            this.tabPage9.Location = new System.Drawing.Point(4, 27);
            this.tabPage9.Name = "tabPage9";
            this.tabPage9.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage9.Size = new System.Drawing.Size(1204, 345);
            this.tabPage9.TabIndex = 8;
            this.tabPage9.Text = "Фурье";
            this.tabPage9.UseVisualStyleBackColor = true;
            // 
            // normalizeCheckBox
            // 
            this.normalizeCheckBox.AutoSize = true;
            this.normalizeCheckBox.Location = new System.Drawing.Point(11, 98);
            this.normalizeCheckBox.Name = "normalizeCheckBox";
            this.normalizeCheckBox.Size = new System.Drawing.Size(326, 22);
            this.normalizeCheckBox.TabIndex = 9;
            this.normalizeCheckBox.Text = "Нормализовывать перед выводом в файл";
            this.normalizeCheckBox.UseVisualStyleBackColor = true;
            // 
            // computeFTButton
            // 
            this.computeFTButton.Location = new System.Drawing.Point(11, 64);
            this.computeFTButton.Name = "computeFTButton";
            this.computeFTButton.Size = new System.Drawing.Size(167, 28);
            this.computeFTButton.TabIndex = 8;
            this.computeFTButton.Text = "Посчитать Фурье";
            this.computeFTButton.UseVisualStyleBackColor = true;
            this.computeFTButton.Click += new System.EventHandler(this.ComputeFTButton_Click);
            // 
            // label29
            // 
            this.label29.AutoSize = true;
            this.label29.Location = new System.Drawing.Point(8, 13);
            this.label29.Name = "label29";
            this.label29.Size = new System.Drawing.Size(137, 18);
            this.label29.TabIndex = 7;
            this.label29.Text = "Папка с замерами";
            // 
            // measurementFolderTextBox
            // 
            this.measurementFolderTextBox.Location = new System.Drawing.Point(11, 34);
            this.measurementFolderTextBox.Name = "measurementFolderTextBox";
            this.measurementFolderTextBox.Size = new System.Drawing.Size(677, 24);
            this.measurementFolderTextBox.TabIndex = 6;
            this.measurementFolderTextBox.Text = "C:\\Users\\user1\\Desktop\\ext_new\\16mm\\bez\\";
            // 
            // tabPage10
            // 
            this.tabPage10.Controls.Add(this.rightButton);
            this.tabPage10.Controls.Add(this.leftButton);
            this.tabPage10.Controls.Add(this.downButton);
            this.tabPage10.Controls.Add(this.upButton);
            this.tabPage10.Controls.Add(this.dyTextBox);
            this.tabPage10.Controls.Add(this.label35);
            this.tabPage10.Controls.Add(this.dxTextBox);
            this.tabPage10.Controls.Add(this.label36);
            this.tabPage10.Controls.Add(this.label34);
            this.tabPage10.Controls.Add(this.resultsTextBox);
            this.tabPage10.Controls.Add(this.send2TextBox);
            this.tabPage10.Controls.Add(this.send2Button);
            this.tabPage10.Controls.Add(this.port2ListBox);
            this.tabPage10.Controls.Add(this.connectPort2Button);
            this.tabPage10.Controls.Add(this.getPorts2Button);
            this.tabPage10.Controls.Add(this.yTextBox);
            this.tabPage10.Controls.Add(this.label33);
            this.tabPage10.Controls.Add(this.xTextBox);
            this.tabPage10.Controls.Add(this.label32);
            this.tabPage10.Controls.Add(this.modeTextBox);
            this.tabPage10.Controls.Add(this.label31);
            this.tabPage10.Controls.Add(this.send1Button);
            this.tabPage10.Location = new System.Drawing.Point(4, 27);
            this.tabPage10.Name = "tabPage10";
            this.tabPage10.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage10.Size = new System.Drawing.Size(1204, 345);
            this.tabPage10.TabIndex = 9;
            this.tabPage10.Text = "gcode test";
            this.tabPage10.UseVisualStyleBackColor = true;
            // 
            // rightButton
            // 
            this.rightButton.Location = new System.Drawing.Point(385, 197);
            this.rightButton.Name = "rightButton";
            this.rightButton.Size = new System.Drawing.Size(75, 23);
            this.rightButton.TabIndex = 50;
            this.rightButton.Text = "RIGHT";
            this.rightButton.UseVisualStyleBackColor = true;
            this.rightButton.Click += new System.EventHandler(this.RightButton_Click);
            // 
            // leftButton
            // 
            this.leftButton.Location = new System.Drawing.Point(222, 197);
            this.leftButton.Name = "leftButton";
            this.leftButton.Size = new System.Drawing.Size(75, 23);
            this.leftButton.TabIndex = 49;
            this.leftButton.Text = "LEFT";
            this.leftButton.UseVisualStyleBackColor = true;
            this.leftButton.Click += new System.EventHandler(this.LeftButton_Click);
            // 
            // downButton
            // 
            this.downButton.Location = new System.Drawing.Point(303, 226);
            this.downButton.Name = "downButton";
            this.downButton.Size = new System.Drawing.Size(75, 23);
            this.downButton.TabIndex = 48;
            this.downButton.Text = "DOWN";
            this.downButton.UseVisualStyleBackColor = true;
            this.downButton.Click += new System.EventHandler(this.DownButton_Click);
            // 
            // upButton
            // 
            this.upButton.Location = new System.Drawing.Point(303, 168);
            this.upButton.Name = "upButton";
            this.upButton.Size = new System.Drawing.Size(75, 23);
            this.upButton.TabIndex = 47;
            this.upButton.Text = "UP";
            this.upButton.UseVisualStyleBackColor = true;
            this.upButton.Click += new System.EventHandler(this.UpButton_Click);
            // 
            // dyTextBox
            // 
            this.dyTextBox.Location = new System.Drawing.Point(63, 291);
            this.dyTextBox.Name = "dyTextBox";
            this.dyTextBox.Size = new System.Drawing.Size(100, 24);
            this.dyTextBox.TabIndex = 46;
            this.dyTextBox.Text = "10";
            this.dyTextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // label35
            // 
            this.label35.AutoSize = true;
            this.label35.Location = new System.Drawing.Point(34, 297);
            this.label35.Name = "label35";
            this.label35.Size = new System.Drawing.Size(23, 18);
            this.label35.TabIndex = 45;
            this.label35.Text = "dy";
            // 
            // dxTextBox
            // 
            this.dxTextBox.Location = new System.Drawing.Point(63, 261);
            this.dxTextBox.Name = "dxTextBox";
            this.dxTextBox.Size = new System.Drawing.Size(100, 24);
            this.dxTextBox.TabIndex = 44;
            this.dxTextBox.Text = "10";
            this.dxTextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // label36
            // 
            this.label36.AutoSize = true;
            this.label36.Location = new System.Drawing.Point(34, 267);
            this.label36.Name = "label36";
            this.label36.Size = new System.Drawing.Size(23, 18);
            this.label36.TabIndex = 43;
            this.label36.Text = "dx";
            // 
            // label34
            // 
            this.label34.AutoSize = true;
            this.label34.Location = new System.Drawing.Point(182, 145);
            this.label34.Name = "label34";
            this.label34.Size = new System.Drawing.Size(42, 18);
            this.label34.TabIndex = 42;
            this.label34.Text = "px py";
            // 
            // resultsTextBox
            // 
            this.resultsTextBox.Location = new System.Drawing.Point(482, 121);
            this.resultsTextBox.Multiline = true;
            this.resultsTextBox.Name = "resultsTextBox";
            this.resultsTextBox.Size = new System.Drawing.Size(201, 167);
            this.resultsTextBox.TabIndex = 41;
            // 
            // send2TextBox
            // 
            this.send2TextBox.Location = new System.Drawing.Point(25, 168);
            this.send2TextBox.Name = "send2TextBox";
            this.send2TextBox.Size = new System.Drawing.Size(91, 24);
            this.send2TextBox.TabIndex = 40;
            this.send2TextBox.Text = "g0 x10 y10";
            // 
            // send2Button
            // 
            this.send2Button.Location = new System.Drawing.Point(25, 212);
            this.send2Button.Name = "send2Button";
            this.send2Button.Size = new System.Drawing.Size(75, 37);
            this.send2Button.TabIndex = 39;
            this.send2Button.Text = "SEND";
            this.send2Button.UseVisualStyleBackColor = true;
            this.send2Button.Click += new System.EventHandler(this.Send2Button_Click);
            // 
            // port2ListBox
            // 
            this.port2ListBox.FormattingEnabled = true;
            this.port2ListBox.ItemHeight = 18;
            this.port2ListBox.Location = new System.Drawing.Point(394, 23);
            this.port2ListBox.Name = "port2ListBox";
            this.port2ListBox.Size = new System.Drawing.Size(146, 58);
            this.port2ListBox.TabIndex = 38;
            // 
            // connectPort2Button
            // 
            this.connectPort2Button.Location = new System.Drawing.Point(264, 53);
            this.connectPort2Button.Name = "connectPort2Button";
            this.connectPort2Button.Size = new System.Drawing.Size(124, 28);
            this.connectPort2Button.TabIndex = 37;
            this.connectPort2Button.Text = "Connect to port";
            this.connectPort2Button.UseVisualStyleBackColor = true;
            this.connectPort2Button.Click += new System.EventHandler(this.ConnectPort2Button_Click);
            // 
            // getPorts2Button
            // 
            this.getPorts2Button.Location = new System.Drawing.Point(264, 23);
            this.getPorts2Button.Name = "getPorts2Button";
            this.getPorts2Button.Size = new System.Drawing.Size(124, 29);
            this.getPorts2Button.TabIndex = 36;
            this.getPorts2Button.Text = "Список портов";
            this.getPorts2Button.UseVisualStyleBackColor = true;
            this.getPorts2Button.Click += new System.EventHandler(this.GetPorts2Button_Click);
            // 
            // yTextBox
            // 
            this.yTextBox.Location = new System.Drawing.Point(109, 80);
            this.yTextBox.Name = "yTextBox";
            this.yTextBox.Size = new System.Drawing.Size(100, 24);
            this.yTextBox.TabIndex = 6;
            this.yTextBox.Text = "y0";
            // 
            // label33
            // 
            this.label33.AutoSize = true;
            this.label33.Location = new System.Drawing.Point(22, 83);
            this.label33.Name = "label33";
            this.label33.Size = new System.Drawing.Size(15, 18);
            this.label33.TabIndex = 5;
            this.label33.Text = "y";
            // 
            // xTextBox
            // 
            this.xTextBox.Location = new System.Drawing.Point(109, 50);
            this.xTextBox.Name = "xTextBox";
            this.xTextBox.Size = new System.Drawing.Size(100, 24);
            this.xTextBox.TabIndex = 4;
            this.xTextBox.Text = "x0";
            // 
            // label32
            // 
            this.label32.AutoSize = true;
            this.label32.Location = new System.Drawing.Point(22, 53);
            this.label32.Name = "label32";
            this.label32.Size = new System.Drawing.Size(15, 18);
            this.label32.TabIndex = 3;
            this.label32.Text = "x";
            // 
            // modeTextBox
            // 
            this.modeTextBox.Location = new System.Drawing.Point(109, 20);
            this.modeTextBox.Name = "modeTextBox";
            this.modeTextBox.Size = new System.Drawing.Size(100, 24);
            this.modeTextBox.TabIndex = 2;
            this.modeTextBox.Text = "g0";
            // 
            // label31
            // 
            this.label31.AutoSize = true;
            this.label31.Location = new System.Drawing.Point(22, 23);
            this.label31.Name = "label31";
            this.label31.Size = new System.Drawing.Size(46, 18);
            this.label31.TabIndex = 1;
            this.label31.Text = "mode";
            // 
            // send1Button
            // 
            this.send1Button.Location = new System.Drawing.Point(25, 115);
            this.send1Button.Name = "send1Button";
            this.send1Button.Size = new System.Drawing.Size(75, 37);
            this.send1Button.TabIndex = 0;
            this.send1Button.Text = "SEND";
            this.send1Button.UseVisualStyleBackColor = true;
            this.send1Button.Click += new System.EventHandler(this.Send1Button_Click);
            // 
            // tabPage11
            // 
            this.tabPage11.Controls.Add(this.moveAlongYCheckBox);
            this.tabPage11.Controls.Add(this.moveAlongXCheckBox);
            this.tabPage11.Controls.Add(this.fTextBox);
            this.tabPage11.Controls.Add(this.label45);
            this.tabPage11.Controls.Add(this.offYTextBox);
            this.tabPage11.Controls.Add(this.label43);
            this.tabPage11.Controls.Add(this.offXTextBox);
            this.tabPage11.Controls.Add(this.label44);
            this.tabPage11.Controls.Add(this.measureAndSaveCheckBox);
            this.tabPage11.Controls.Add(this.button2);
            this.tabPage11.Controls.Add(this.collectData2Button);
            this.tabPage11.Controls.Add(this.nyTextBox);
            this.tabPage11.Controls.Add(this.label41);
            this.tabPage11.Controls.Add(this.nxTextBox);
            this.tabPage11.Controls.Add(this.label42);
            this.tabPage11.Controls.Add(this.y1TextBox);
            this.tabPage11.Controls.Add(this.label39);
            this.tabPage11.Controls.Add(this.x1TextBox);
            this.tabPage11.Controls.Add(this.label40);
            this.tabPage11.Controls.Add(this.y0TextBox);
            this.tabPage11.Controls.Add(this.label37);
            this.tabPage11.Controls.Add(this.x0TextBox);
            this.tabPage11.Controls.Add(this.label38);
            this.tabPage11.Location = new System.Drawing.Point(4, 27);
            this.tabPage11.Name = "tabPage11";
            this.tabPage11.Size = new System.Drawing.Size(1204, 345);
            this.tabPage11.TabIndex = 10;
            this.tabPage11.Text = "ScanAcoustic";
            this.tabPage11.UseVisualStyleBackColor = true;
            // 
            // moveAlongYCheckBox
            // 
            this.moveAlongYCheckBox.AutoSize = true;
            this.moveAlongYCheckBox.Location = new System.Drawing.Point(178, 135);
            this.moveAlongYCheckBox.Name = "moveAlongYCheckBox";
            this.moveAlongYCheckBox.Size = new System.Drawing.Size(118, 22);
            this.moveAlongYCheckBox.TabIndex = 28;
            this.moveAlongYCheckBox.Text = "Move Along Y";
            this.moveAlongYCheckBox.UseVisualStyleBackColor = true;
            // 
            // moveAlongXCheckBox
            // 
            this.moveAlongXCheckBox.AutoSize = true;
            this.moveAlongXCheckBox.Checked = true;
            this.moveAlongXCheckBox.CheckState = System.Windows.Forms.CheckState.Checked;
            this.moveAlongXCheckBox.Location = new System.Drawing.Point(178, 107);
            this.moveAlongXCheckBox.Name = "moveAlongXCheckBox";
            this.moveAlongXCheckBox.Size = new System.Drawing.Size(119, 22);
            this.moveAlongXCheckBox.TabIndex = 27;
            this.moveAlongXCheckBox.Text = "Move Along X";
            this.moveAlongXCheckBox.UseVisualStyleBackColor = true;
            // 
            // fTextBox
            // 
            this.fTextBox.Location = new System.Drawing.Point(166, 26);
            this.fTextBox.Name = "fTextBox";
            this.fTextBox.Size = new System.Drawing.Size(52, 24);
            this.fTextBox.TabIndex = 26;
            this.fTextBox.Text = "200";
            this.fTextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // label45
            // 
            this.label45.AutoSize = true;
            this.label45.Location = new System.Drawing.Point(148, 32);
            this.label45.Name = "label45";
            this.label45.Size = new System.Drawing.Size(12, 18);
            this.label45.TabIndex = 25;
            this.label45.Text = "f";
            // 
            // offYTextBox
            // 
            this.offYTextBox.Location = new System.Drawing.Point(69, 265);
            this.offYTextBox.Name = "offYTextBox";
            this.offYTextBox.Size = new System.Drawing.Size(52, 24);
            this.offYTextBox.TabIndex = 24;
            this.offYTextBox.Text = "0";
            this.offYTextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // label43
            // 
            this.label43.AutoSize = true;
            this.label43.Location = new System.Drawing.Point(23, 268);
            this.label43.Name = "label43";
            this.label43.Size = new System.Drawing.Size(40, 18);
            this.label43.TabIndex = 23;
            this.label43.Text = "off_y";
            // 
            // offXTextBox
            // 
            this.offXTextBox.Location = new System.Drawing.Point(69, 232);
            this.offXTextBox.Name = "offXTextBox";
            this.offXTextBox.Size = new System.Drawing.Size(52, 24);
            this.offXTextBox.TabIndex = 22;
            this.offXTextBox.Text = "0";
            this.offXTextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // label44
            // 
            this.label44.AutoSize = true;
            this.label44.Location = new System.Drawing.Point(23, 238);
            this.label44.Name = "label44";
            this.label44.Size = new System.Drawing.Size(40, 18);
            this.label44.TabIndex = 21;
            this.label44.Text = "off_x";
            // 
            // measureAndSaveCheckBox
            // 
            this.measureAndSaveCheckBox.AutoSize = true;
            this.measureAndSaveCheckBox.Checked = true;
            this.measureAndSaveCheckBox.CheckState = System.Windows.Forms.CheckState.Checked;
            this.measureAndSaveCheckBox.Location = new System.Drawing.Point(435, 40);
            this.measureAndSaveCheckBox.Name = "measureAndSaveCheckBox";
            this.measureAndSaveCheckBox.Size = new System.Drawing.Size(183, 22);
            this.measureAndSaveCheckBox.TabIndex = 20;
            this.measureAndSaveCheckBox.Text = "Измерять и сохранять";
            this.measureAndSaveCheckBox.UseVisualStyleBackColor = true;
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(329, 83);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(139, 37);
            this.button2.TabIndex = 19;
            this.button2.Text = "Сбор данных V2";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // collectData2Button
            // 
            this.collectData2Button.Location = new System.Drawing.Point(275, 32);
            this.collectData2Button.Name = "collectData2Button";
            this.collectData2Button.Size = new System.Drawing.Size(122, 37);
            this.collectData2Button.TabIndex = 19;
            this.collectData2Button.Text = "Сбор данных";
            this.collectData2Button.UseVisualStyleBackColor = true;
            this.collectData2Button.Click += new System.EventHandler(this.CollectData2Button_Click);
            // 
            // nyTextBox
            // 
            this.nyTextBox.Location = new System.Drawing.Point(52, 185);
            this.nyTextBox.Name = "nyTextBox";
            this.nyTextBox.Size = new System.Drawing.Size(52, 24);
            this.nyTextBox.TabIndex = 18;
            this.nyTextBox.Text = "1";
            this.nyTextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // label41
            // 
            this.label41.AutoSize = true;
            this.label41.Location = new System.Drawing.Point(23, 188);
            this.label41.Name = "label41";
            this.label41.Size = new System.Drawing.Size(23, 18);
            this.label41.TabIndex = 17;
            this.label41.Text = "ny";
            // 
            // nxTextBox
            // 
            this.nxTextBox.Location = new System.Drawing.Point(52, 152);
            this.nxTextBox.Name = "nxTextBox";
            this.nxTextBox.Size = new System.Drawing.Size(52, 24);
            this.nxTextBox.TabIndex = 16;
            this.nxTextBox.Text = "200";
            this.nxTextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // label42
            // 
            this.label42.AutoSize = true;
            this.label42.Location = new System.Drawing.Point(23, 158);
            this.label42.Name = "label42";
            this.label42.Size = new System.Drawing.Size(23, 18);
            this.label42.TabIndex = 15;
            this.label42.Text = "nx";
            // 
            // y1TextBox
            // 
            this.y1TextBox.Location = new System.Drawing.Point(52, 122);
            this.y1TextBox.Name = "y1TextBox";
            this.y1TextBox.Size = new System.Drawing.Size(52, 24);
            this.y1TextBox.TabIndex = 14;
            this.y1TextBox.Text = "0";
            this.y1TextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // label39
            // 
            this.label39.AutoSize = true;
            this.label39.Location = new System.Drawing.Point(23, 125);
            this.label39.Name = "label39";
            this.label39.Size = new System.Drawing.Size(23, 18);
            this.label39.TabIndex = 13;
            this.label39.Text = "y1";
            // 
            // x1TextBox
            // 
            this.x1TextBox.Location = new System.Drawing.Point(52, 89);
            this.x1TextBox.Name = "x1TextBox";
            this.x1TextBox.Size = new System.Drawing.Size(52, 24);
            this.x1TextBox.TabIndex = 12;
            this.x1TextBox.Text = "100";
            this.x1TextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // label40
            // 
            this.label40.AutoSize = true;
            this.label40.Location = new System.Drawing.Point(23, 95);
            this.label40.Name = "label40";
            this.label40.Size = new System.Drawing.Size(23, 18);
            this.label40.TabIndex = 11;
            this.label40.Text = "x1";
            // 
            // y0TextBox
            // 
            this.y0TextBox.Location = new System.Drawing.Point(52, 59);
            this.y0TextBox.Name = "y0TextBox";
            this.y0TextBox.Size = new System.Drawing.Size(52, 24);
            this.y0TextBox.TabIndex = 10;
            this.y0TextBox.Text = "0";
            this.y0TextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // label37
            // 
            this.label37.AutoSize = true;
            this.label37.Location = new System.Drawing.Point(23, 62);
            this.label37.Name = "label37";
            this.label37.Size = new System.Drawing.Size(23, 18);
            this.label37.TabIndex = 9;
            this.label37.Text = "y0";
            // 
            // x0TextBox
            // 
            this.x0TextBox.Location = new System.Drawing.Point(52, 26);
            this.x0TextBox.Name = "x0TextBox";
            this.x0TextBox.Size = new System.Drawing.Size(52, 24);
            this.x0TextBox.TabIndex = 8;
            this.x0TextBox.Text = "0";
            this.x0TextBox.TextChanged += new System.EventHandler(this.NumericTextField_TextChanged);
            // 
            // label38
            // 
            this.label38.AutoSize = true;
            this.label38.Location = new System.Drawing.Point(23, 32);
            this.label38.Name = "label38";
            this.label38.Size = new System.Drawing.Size(23, 18);
            this.label38.TabIndex = 7;
            this.label38.Text = "x0";
            // 
            // progressBar1
            // 
            this.progressBar1.Location = new System.Drawing.Point(3, 384);
            this.progressBar1.Name = "progressBar1";
            this.progressBar1.Size = new System.Drawing.Size(692, 23);
            this.progressBar1.TabIndex = 1;
            // 
            // timer1
            // 
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick_1);
            // 
            // timer2
            // 
            this.timer2.Tick += new System.EventHandler(this.timer2_Tick);
            // 
            // PS5000ABlockForm
            // 
            this.ClientSize = new System.Drawing.Size(1277, 523);
            this.Controls.Add(this.progressBar1);
            this.Controls.Add(this.tabControl1);
            this.Name = "PS5000ABlockForm";
            this.tabControl1.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage1.PerformLayout();
            this.tabPage3.ResumeLayout(false);
            this.tabPage3.PerformLayout();
            this.tabPage2.ResumeLayout(false);
            this.tabPage2.PerformLayout();
            this.tabPage4.ResumeLayout(false);
            this.tabPage4.PerformLayout();
            this.tabPage5.ResumeLayout(false);
            this.tabPage5.PerformLayout();
            this.tabPage6.ResumeLayout(false);
            this.flowLayoutPanel1.ResumeLayout(false);
            this.flowLayoutPanel1.PerformLayout();
            this.tabPage7.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.chart1)).EndInit();
            this.tabPage8.ResumeLayout(false);
            this.tabPage9.ResumeLayout(false);
            this.tabPage9.PerformLayout();
            this.tabPage10.ResumeLayout(false);
            this.tabPage10.PerformLayout();
            this.tabPage11.ResumeLayout(false);
            this.tabPage11.PerformLayout();
            this.ResumeLayout(false);

        }

        private void CollectDataButton_Click(object sender, EventArgs e)
        {
            string dir = filePathTextBox.Text;
            if (dir[dir.Length - 1] != '\\')
            {
                dir = String.Concat(dir, "\\");
            }

            Directory.CreateDirectory(dir);

            timer1.Enabled = true;

            uint samples1 = uint.Parse(stepsBeforeTextBox.Text);
            uint samples2 = uint.Parse(stepsAfterTextBox.Text);
            uint count_avg = uint.Parse(averagingsTextBox.Text);

            masA = new long[samples1 + samples2];
            _arrA = new double[samples1 + samples2];
            _all = int.Parse(averagingsTextBox.Text);

            ushort RANGE_ = _input_ranges[comboRangeA.SelectedIndex];
 
            switch (oscilloscopeSwitch.SelectedItem.ToString())
            {
                case "Picoscope":
                    CollectData(samples1, samples2, count_avg, RANGE_, masA, _arrA, _all);
                    break;
                case "OWON":
                    CollectData_OWON();
                    break;
            }
             
            Application.DoEvents();

            if (suppressStartCheckBox.Checked)
            {
                SuppressSpikes(_arrA);
            }

            bool _reason =
                suppressStartCheckBox.Checked ||
                applyAutoRunAvgCheckBox.Checked ||
                autoEliminateAvgCheckBox.Checked;

            if (_reason)
            {
                if (suppressStartCheckBox.Checked)
                {
                    SuppressSpikes(_arrA);
                }
                if (applyAutoRunAvgCheckBox.Checked)
                {
                    RunAvg(ref _arrA, int.Parse(runAvgTextBox.Text));
                };
                if (autoEliminateAvgCheckBox.Checked)
                {
                    NoOffset(_arrA);
                };

            }

            if (useTimeFilterCheckBox.Checked)
            {
                if (useTimeFilterTextBox.Text.Length > 0)
                {
                    Complex[] filtr = LoadFromFileC(useTimeFilterTextBox.Text);
                    _arrA = FuncMult(_arrA, _oscilloscope_timestep, -_oscilloscope_timestep * double.Parse(stepsBeforeTextBox.Text), filtr);
                }


            }


            if (useFreqFilterCheckBox.Checked)
            {
                if (useFreqFilterTextBox.Text.Length > 0)
                {
                    Complex[] f = FurieTransf(_arrA, _oscilloscope_timestep, -_oscilloscope_timestep * double.Parse(stepsBeforeTextBox.Text),
                         double.Parse(FTBottomTextBox.Text), double.Parse(FTStepTextBox.Text), int.Parse(FTStepsNumberTextBox.Text));
                    Complex[] filtr = LoadFromFileC(useFreqFilterTextBox.Text);
                    f = FuncMult(f, double.Parse(FTStepTextBox.Text), double.Parse(FTBottomTextBox.Text), filtr);
                    Complex[] restored = FurieTransfReverse(f, _oscilloscope_timestep, -_oscilloscope_timestep * double.Parse(stepsBeforeTextBox.Text), _arrA.Length, double.Parse(FTBottomTextBox.Text), double.Parse(FTStepTextBox.Text));
                    for (int k1 = 0; k1 < restored.Length; k1++)
                    {
                        _arrA[k1] = restored[k1].Real * 2;//важно делать умножение на 2 так как интеграл по полубесконечному промежутку
                    }
                }


            }

            if (autoEliminateAvgCheckBox.Checked)
            {
                //NoOffset(_arrA);
            };

            timer1.Enabled = false;
            dir = String.Concat(dir, fileNameTextBox.Text);
            Save2File(dir, _arrA);
            if (_stop_flag)
            {
                _save = 0; _stop_flag = false;
            }

        }

        private void timer1_Tick_1(object sender, EventArgs e)
        {
            Application.DoEvents();
        }

        private void ConnectAsReceiverButton_Click(object sender, EventArgs e)
        {
            try
            {
                Switch1.OUTRelay(int.Parse(sensorTextBox.Text));
                for (int k1 = 0; k1 < Switch1.Kirill_DATA.Length; k1++)
                {
                    string ssss = Convert.ToString(Switch1.Kirill_DATA[k1], 2);
                    ssss = Switch1.Kirill_DATA[k1].ToString();
                    textBoxUnitInfo.AppendText(ssss + Environment.NewLine);

                }

                Switch1.sendKIRILL();
                Thread.Sleep(500);
                string txt = Switch1.GetAcceptedKiril();
                textBoxUnitInfo.AppendText(txt + Environment.NewLine);
                this.Text = txt;

            }
            catch (Exception)
            {
                MessageBox.Show("Не удалось послать команду");
            }
        }

        private void DisconnectButton_Click(object sender, EventArgs e)
        {
            try
            {
                Switch1.OFFRelay(int.Parse(sensorTextBox.Text));
                Switch1.sendKIRILL();
                Thread.Sleep(500);
                string txt = Switch1.GetAcceptedKiril();
                textBoxUnitInfo.AppendText(txt + "\n");
                this.Text = txt;
            }
            catch (Exception)
            {
                MessageBox.Show("Не удалось послать команду");
            }
        }

        private void ConnectAsSourceButton_Click(object sender, EventArgs e)
        {
            try
            {
                Switch1.INRelay(int.Parse(sensorTextBox.Text));

                for (int k1 = 0; k1 < Switch1.Kirill_DATA.Length; k1++)
                {
                    string ssss = Convert.ToString(Switch1.Kirill_DATA[k1], 2);
                    ssss = Switch1.Kirill_DATA[k1].ToString();
                    textBoxUnitInfo.AppendText(ssss + Environment.NewLine);
                }

                Switch1.sendKIRILL();
                Thread.Sleep(500);
                string txt = Switch1.GetAcceptedKiril();
                textBoxUnitInfo.AppendText(txt + Environment.NewLine);
                this.Text = txt;
            }
            catch (Exception)
            {
                MessageBox.Show("Не удалось послать команду");
            }
        }

        private void GetPortsButton_Click(object sender, EventArgs e)
        {
            names_ = SerialPort.GetPortNames();
            portListBox.Items.Clear();
            portListBox.Items.AddRange(names_);

            if (portListBox.Items.Count > 0)
            {
                portListBox.SelectedIndex = portListBox.Items.Count - 1;
            }
        }

        private void ConnectPortButton_Click(object sender, EventArgs e)
        {
            if (Switch1 == null)
            {
                Switch1 = new Switch();
                string com_id = portListBox.Items[portListBox.SelectedIndex].ToString();
                string com_id2 = com_id.Substring(3);
                int c_id = int.Parse(com_id2);

                Switch1.InitKirill(int.Parse(layersTextBox.Text), portListBox.SelectedIndex);
                Thread.Sleep(500);
                string txt = Switch1.GetAcceptedKiril();
                textBoxUnitInfo.AppendText(txt + "\n");
                this.Text = txt;
                _switch_connected = true;
                Switch1.sendKIRILL();
                Thread.Sleep(500);
                txt = Switch1.GetAcceptedKiril();
                textBoxUnitInfo.AppendText(txt + "\n");
                connectPortButton.Text = "Отключить";
            }
            else
            {
                Switch1.ClosePort_();
                connectPortButton.Text = "Подключить";
                Switch1 = null;
            }

        }

        private void StopButton_Click(object sender, EventArgs e)
        {

            _stop_flag = true;
            timer1.Enabled = false;
        }

        private void ApplyRunAvgButton_Click(object sender, EventArgs e)
        {
            RunAvg(ref _arrA, int.Parse(runAvgTextBox.Text));
            string path = String.Concat(@"C:\Temp\", DateTime.Now.ToString().Replace(':', '_'), "TempCapture.txt");

            Save2File(path, _arrA);

        }

        private void tabPage6_Click(object sender, EventArgs e)
        {
            if (_arrA != null)
            {
                // Visualase(_arrA);
            }
        }

        private void tabControl1_SelectedIndexChanged(object sender, EventArgs e)
        {
            //if (_arrA != null)
            //{
            //    Visualase(_arrA);
            //}
        }

        private void button11_Click(object sender, EventArgs e)
        {
            if (_switch_connected)
            {

                uint samples1 = uint.Parse(stepsBeforeTextBox.Text);
                uint samples2 = uint.Parse(stepsAfterTextBox.Text);
                uint count_avg = uint.Parse(averagingsTextBox.Text);

                masA = new long[samples1 + samples2];
                _arrA = new double[samples1 + samples2];
                _all = int.Parse(averagingsTextBox.Text);

                ushort RANGE_ = _input_ranges[comboRangeA.SelectedIndex];


                string dir = filePathTextBox.Text;

                if (dir[dir.Length - 1] != '\\')
                    dir = String.Concat(dir, "\\");

                Directory.CreateDirectory(dir);

                int step_i = int.Parse(saveEveryTextBox.Text);

                //вывод файла с временами
                if (saveTimesCheckBox.Checked)
                {
                    int l = int.Parse(stepsAfterTextBox.Text) + int.Parse(stepsBeforeTextBox.Text);
                    double[] dt = new double[l];
                    double n0 = double.Parse(stepsBeforeTextBox.Text);
                    for (int i = 0; i < l; i += step_i)
                    {
                        dt[i] = _oscilloscope_timestep * i - _oscilloscope_timestep * n0;
                    }
                    string fn = "times.txt";
                    Save2File(String.Concat(dir, fn), dt, step_i);
                }

                //==============================================================


                for (int i = 0; i < receiversCheckedListBox.CheckedIndices.Count; i++)
                {
                    int j = receiversCheckedListBox.CheckedIndices[i];

                    {
                        string txt = Switch1.GetAcceptedKiril();
                        textBoxUnitInfo.AppendText(txt + "\n");
                        this.Text = txt;

                        Switch1.OFFRelay(j);
                        Switch1.OUTRelay(j);
                        Switch1.sendKIRILL();

                        // Switch1.SendCmd(0, j);
                        while (Switch1.port.BytesToRead == 0) { Thread.Sleep(50); };
                        txt = Switch1.GetAcceptedKiril();
                        textBoxUnitInfo.AppendText(txt + "\n");
                        this.Text = txt;
                        // Thread.Sleep(500);// подождем пока напряжение устаканится на ПЭ
                        for (int k = 0; k < sourcesCheckedListBox.CheckedIndices.Count; k++)
                        {

                            int m = sourcesCheckedListBox.CheckedIndices[k];
                            if ((m != j))

                            {

                                txt = Switch1.GetAcceptedKiril();
                                textBoxUnitInfo.AppendText(txt + "\n");
                                this.Text = txt;

                                Switch1.OFFRelay(m);
                                Switch1.INRelay(m);
                                Switch1.sendKIRILL();

                                //Switch1.SendCmd(1, m);
                                while (Switch1.port.BytesToRead == 0) { Thread.Sleep(50); };
                                txt = Switch1.GetAcceptedKiril();
                                textBoxUnitInfo.AppendText(txt + "\n");
                                this.Text = txt;

                                timer1.Enabled = true;

                                CollectData(samples1, samples2, count_avg, RANGE_, masA, _arrA, _all);
                                Application.DoEvents();

                                if (saveRawCheckBox.Checked)
                                {
                                    dir = String.Concat(filePathTextBox.Text, CODES[j], "\\");
                                    Directory.CreateDirectory(dir);
                                    string fn_ = String.Concat("raw_", CODES[j], "2", CODES[m], ".txt");
                                    Save2File(String.Concat(dir, fn_), _arrA, step_i);
                                }

                                bool _reason = suppressStartCheckBox.Checked || applyAutoRunAvgCheckBox.Checked || autoEliminateAvgCheckBox.Checked;
                                if (_reason)
                                {
                                    if (suppressStartCheckBox.Checked)
                                    {
                                        SuppressSpikes(_arrA);
                                    }
                                    if (applyAutoRunAvgCheckBox.Checked)
                                    {
                                        RunAvg(ref _arrA, int.Parse(runAvgTextBox.Text));
                                    };
                                    if (autoEliminateAvgCheckBox.Checked)
                                    {
                                        NoOffset(_arrA);
                                    };
                                    timer1.Enabled = false;
                                    if (_stop_flag)
                                    {
                                        _save = 0; _stop_flag = false;
                                    }
                                }
                                //============================================================
                                //вставить применение фильтра
                                if (useTimeFilterCheckBox.Checked)
                                {
                                    if (useTimeFilterTextBox.Text.Length > 0)
                                    {
                                        Complex[] filtr = LoadFromFileC(useTimeFilterTextBox.Text);
                                        _arrA = FuncMult(_arrA, _oscilloscope_timestep, -_oscilloscope_timestep * double.Parse(stepsBeforeTextBox.Text), filtr);
                                    }

                                    dir = String.Concat(filePathTextBox.Text, CODES[j], "\\");
                                    Directory.CreateDirectory(dir);
                                    string fn_ = String.Concat("afterf1_", CODES[j], "2", CODES[m], ".txt");
                                    Save2File(String.Concat(dir, fn_), _arrA, step_i);
                                }





                                if (useFreqFilterCheckBox.Checked)
                                {
                                    if (useFreqFilterTextBox.Text.Length > 0)
                                    {
                                        Complex[] f = FurieTransf(_arrA, _oscilloscope_timestep, -_oscilloscope_timestep * double.Parse(stepsBeforeTextBox.Text),
                                             double.Parse(FTBottomTextBox.Text), double.Parse(FTStepTextBox.Text), int.Parse(FTStepsNumberTextBox.Text));
                                        Complex[] filtr = LoadFromFileC(useFreqFilterTextBox.Text);
                                        f = FuncMult(f, double.Parse(FTStepTextBox.Text), double.Parse(FTBottomTextBox.Text), filtr);
                                        Complex[] restored = FurieTransfReverse(f, _oscilloscope_timestep, -_oscilloscope_timestep * double.Parse(stepsBeforeTextBox.Text), _arrA.Length, double.Parse(FTBottomTextBox.Text), double.Parse(FTStepTextBox.Text));
                                        for (int k1 = 0; k1 < restored.Length; k1++)
                                        {
                                            _arrA[k1] = restored[k1].Real * 2;//важно делать умножение на 2 так как интеграл по полубесконечному промежутку
                                        }
                                    }
                                    dir = String.Concat(filePathTextBox.Text, CODES[j], "\\");
                                    Directory.CreateDirectory(dir);
                                    string fn_ = String.Concat("afterf2_", CODES[j], "2", CODES[m], ".txt");
                                    Save2File(String.Concat(dir, fn_), _arrA, step_i);

                                }



                                Visualase(_arrA);
                                ///===========================================================

                                dir = String.Concat(filePathTextBox.Text, CODES[j], "\\");
                                Directory.CreateDirectory(dir);
                                string fn = String.Concat(CODES[j], "2", CODES[m], ".txt");
                                Save2File(String.Concat(dir, fn), _arrA, step_i);
                                if (computeFTCheckBox.Checked)
                                {
                                    Complex[] f = FurieTransf(_arrA, _oscilloscope_timestep, -_oscilloscope_timestep * double.Parse(stepsBeforeTextBox.Text), 1 * double.Parse(FTBottomTextBox.Text), double.Parse(FTStepTextBox.Text), int.Parse(FTStepsNumberTextBox.Text));
                                    Save2File(String.Concat(dir, "f_", fn), f, step_i);
                                    if (computeModuleCheckBox.Checked)
                                    {
                                        double[] abs_f = new double[f.Length];
                                        for (int k1 = 0; k1 < f.Length; k1++)
                                        {
                                            abs_f[k1] = f[k1].Magnitude;
                                        }
                                        Save2File(String.Concat(dir, "abs_f_", fn), abs_f, step_i);
                                    }

                                }
                            }
                        }
                    }
                }
            }
        }

        private void BuildDifferencesButton_Click(object sender, EventArgs e)
        {
            try
            {
                //Создаём или перезаписываем существующий файл
                string Path_ = diffFolderTextBox.Text;

                if (Path_[Path_.Length - 1] != '\\')
                    Path_ = String.Concat(Path_, "\\");


                Directory.CreateDirectory(Path_);

                StreamWriter sw = File.CreateText(String.Concat(Path_, "info.txt"));

                //Записываем текст в поток файла
                sw.WriteLine(differenceInfoTextBox.Text);

                //Закрываем файл
                sw.Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message);
            }
            for (int i = 0; i < receiversCheckedListBox.CheckedIndices.Count; i++)
            {
                int j = receiversCheckedListBox.CheckedIndices[i];

                for (int k = 0; k < sourcesCheckedListBox.CheckedIndices.Count; k++)
                {
                    int m = sourcesCheckedListBox.CheckedIndices[k];

                    if (m != j)
                    {
                        string Path_ = noDefectFolderTextBox.Text;

                        if (Path_[Path_.Length - 1] != '\\')
                        {
                            Path_ = String.Concat(Path_, "\\");
                        }

                        Directory.CreateDirectory(Path_);
                        string dir1 = String.Concat(Path_, CODES[j], "\\");

                        Path_ = defectFolderTextBox.Text;
                        if (Path_[Path_.Length - 1] != '\\')
                            Path_ = String.Concat(Path_, "\\");

                        Directory.CreateDirectory(Path_);
                        string dir2 = String.Concat(Path_, CODES[j], "\\");

                        Path_ = diffFolderTextBox.Text;
                        if (Path_[Path_.Length - 1] != '\\')
                            Path_ = String.Concat(Path_, "\\");

                        Directory.CreateDirectory(Path_);
                        string dir3 = String.Concat(Path_, CODES[j], "\\");
                        string fn = String.Concat(CODES[j], "2", CODES[m], ".txt");
                        StreamReader R1 = new StreamReader(String.Concat(dir1, fn));
                        StreamReader R2 = new StreamReader(String.Concat(dir2, fn));
                        int l1 = int.Parse(stepsAfterTextBox.Text) + int.Parse(stepsBeforeTextBox.Text);
                        int l = l1;
                        Directory.CreateDirectory(dir3);

                        using (StreamWriter Writer = new StreamWriter(String.Concat(dir3, fn)))
                        {

                            double[] buf = new double[l];

                            for (int n = 0; n < l; n++)
                            {
                                string s1 = R1.ReadLine().Replace('.', ',');
                                double d1 = double.Parse(s1);
                                string s2 = R2.ReadLine().Replace('.', ','); ;
                                double d2 = double.Parse(s2);
                                buf[n] = d1 - d2;
                            }
                            if (suppressToCheckBox.Checked)
                            {
                                int pod = int.Parse(suppressToTextBox.Text);

                                for (int n = 0; n < pod; n++)
                                    buf[n] = 0;

                            }
                            if (normBeforeOutputCheckBox.Checked)
                            {
                                double maxc_ = 0;

                                for (int n = 0; n < l; n++)
                                {
                                    double a = Math.Abs(buf[n]);
                                    if (a > maxc_)
                                        maxc_ = a;

                                }

                                for (int n = 0; n < l; n++)
                                    buf[n] = buf[n] / maxc_;

                            }

                            if (applyFreqFilterCheckBox.Checked)
                            {

                                if (useFreqFilterTextBox.Text.Length > 0)
                                {


                                    _oscilloscope_timestep = double.Parse(timebaseTextBox.Text);
                                    if (_oscilloscope_timestep < 4.0)
                                        throw new Exception();

                                    _oscilloscope_timestep = (_oscilloscope_timestep - 3.0) / 62500000.0;

                                    Complex[] f = FurieTransf(buf, _oscilloscope_timestep, -_oscilloscope_timestep * double.Parse(stepsBeforeTextBox.Text),
                                         double.Parse(FTBottomTextBox.Text), double.Parse(FTStepTextBox.Text), int.Parse(FTStepsNumberTextBox.Text));
                                    Complex[] filtr = LoadFromFileC(useFreqFilterTextBox.Text);

                                    f = FuncMult(f, double.Parse(FTStepTextBox.Text), double.Parse(FTBottomTextBox.Text), filtr);

                                    Complex[] restored = FurieTransfReverse(f, _oscilloscope_timestep, -_oscilloscope_timestep * double.Parse(stepsBeforeTextBox.Text),
                                        buf.Length, double.Parse(FTBottomTextBox.Text), double.Parse(FTStepTextBox.Text));

                                    for (int k1 = 0; k1 < restored.Length; k1++)
                                        buf[k1] = restored[k1].Real * 2;//важно делать умножение на 2 так как интеграл по полубесконечному промежутку

                                }
                                Directory.CreateDirectory(dir3);
                                fn = String.Concat(CODES[j], "2", CODES[m], ".txt");


                            }


                            for (int n = 0; n < l; n++)
                            {
                                Writer.WriteLine(buf[n].ToString().Replace(',', '.'));
                            }
                            Writer.Flush();
                            Writer.Close();
                        }
                    }
                }
            }
        }



        private void InvertSensorsButton_Click(object sender, EventArgs e)
        {

            for (int k = 0; k < receiversCheckedListBox.Items.Count; k++)
                receiversCheckedListBox.SetItemChecked(k, !receiversCheckedListBox.GetItemChecked(k));

            for (int k = 0; k < sourcesCheckedListBox.Items.Count; k++)
                sourcesCheckedListBox.SetItemChecked(k, !sourcesCheckedListBox.GetItemChecked(k));

        }

        private void SelectFolderButton_Click(object sender, EventArgs e)
        {
            if (DialogResult.OK == folderBrowserDialog1.ShowDialog())
                filePathTextBox.Text = folderBrowserDialog1.SelectedPath;
        }

        private void button15_Click(object sender, EventArgs e)
        {
            button15.Text = "sasa";
            if (fileMFT == null)
                fileMFT = new FileMFT();

            fileMFT.SetName(@"C:\IMMI\Team\Bareiko\2020\ZAMERI\___________________TYOMICH_OMNOMNOM\10mm_257_54_razn\B\B2E.mftd");
            fileMFT.LoadMFTD();
            fileMFT.SetName(@"C:\IMMI\Team\Bareiko\2020\ZAMERI\___________________TYOMICH_OMNOMNOM\10mm_257_54_razn\B\test_z.txt");
            fileMFT.SaveTXT();

        }

        private void DisconnectAllButton_Click(object sender, EventArgs e)
        {
            for (int i = 0; i < Switch1.Kirill_DATA.Length; i++)
                Switch1.Kirill_DATA[i] = 0;

            for (int k1 = 0; k1 < Switch1.Kirill_DATA.Length; k1++)
            {
                string ssss = Convert.ToString(Switch1.Kirill_DATA[k1], 2);
                ssss = Switch1.Kirill_DATA[k1].ToString();
                textBoxUnitInfo.AppendText(ssss + Environment.NewLine);
            }

            Switch1.sendKIRILL();

            Thread.Sleep(500);
            string txt = Switch1.GetAcceptedKiril();

            textBoxUnitInfo.AppendText(txt + Environment.NewLine);
            this.Text = txt;
        }

        private void ComputeFTButton_Click(object sender, EventArgs e)
        {
            _oscilloscope_timestep = double.Parse(timebaseTextBox.Text);
            if (_oscilloscope_timestep < 4.0)
                throw new Exception();

            _oscilloscope_timestep = (_oscilloscope_timestep - 3.0) / 62500000.0;

            //Написать  метод для считающий фурье
            for (int i = 0; i < receiversCheckedListBox.CheckedIndices.Count; i++)
            {
                int j = receiversCheckedListBox.CheckedIndices[i];

                for (int k = 0; k < sourcesCheckedListBox.CheckedIndices.Count; k++)
                {
                    int m = sourcesCheckedListBox.CheckedIndices[k];

                    if (m != j)
                    {

                        // длинна замера
                        int l1 = int.Parse(stepsAfterTextBox.Text) + int.Parse(stepsBeforeTextBox.Text);
                        int l = l1;
                        string Path_ = measurementFolderTextBox.Text;

                        if (Path_[Path_.Length - 1] != '\\')
                            Path_ = String.Concat(Path_, "\\");


                        //Открываем файл для ввода
                        string fn = String.Concat(CODES[j], "2", CODES[m], ".txt");
                        string path_ = String.Concat(Path_, CODES[j], '\\', fn);
                        StreamReader R1 = new StreamReader(path_);
                        double[] buf = new double[l];

                        for (int n = 0; n < l; n++)
                        {
                            string s1 = R1.ReadLine().Replace('.', ',');
                            buf[n] = double.Parse(s1); ;
                        }

                        //Преобразования фурье 
                        Complex[] f = FurieTransf(buf, _oscilloscope_timestep, -_oscilloscope_timestep * double.Parse(stepsBeforeTextBox.Text),
                             double.Parse(FTBottomTextBox.Text), double.Parse(FTStepTextBox.Text), int.Parse(FTStepsNumberTextBox.Text));

                        //Открываем файл для вывода
                        path_ = String.Concat(Path_, CODES[j], '\\', "furie_", fn);

                        using (StreamWriter Writer = new StreamWriter(path_))
                        {
                            for (int n = 0; n < f.Length; n++)
                                Writer.WriteLine(f[n].ToString().Replace(',', '.').Replace(". ", ", "));

                            Writer.Flush();
                            Writer.Close();
                        }

                        //Открываем файл для вывода
                        path_ = String.Concat(Path_, CODES[j], '\\', "abs_furie_", fn);

                        using (StreamWriter Writer = new StreamWriter(path_))
                        {
                            for (int n = 0; n < f.Length; n++)
                                Writer.WriteLine(f[n].Magnitude.ToString().Replace(',', '.').Replace(". ", ", "));

                            Writer.Flush();
                            Writer.Close();
                        }

                    }
                }
            }
        }

        private void Send1Button_Click(object sender, EventArgs e)
        {
            string cmd = String.Concat(modeTextBox.Text, ' ', xTextBox.Text, ' ', yTextBox.Text);
            Switch1.SendCmd(cmd);
        }

        private void GetPorts2Button_Click(object sender, EventArgs e)
        {
            names_ = SerialPort.GetPortNames();
            port2ListBox.Items.Clear();
            port2ListBox.Items.AddRange(names_);

            if (port2ListBox.Items.Count > 0)
                port2ListBox.SelectedIndex = portListBox.Items.Count - 1;

        }

        private void ConnectPort2Button_Click(object sender, EventArgs e)
        {
            if (!_switch_connected)
            {

                Switch1 = new Switch();

                string com_id = port2ListBox.Items[port2ListBox.SelectedIndex].ToString();

                Switch1.OpenPort(com_id, 115200);
                _switch_connected = true;
                timer2.Interval = 50;
                timer2.Start();

                string sss = String.Concat("px = ", _cnc_x.ToString(), " py = ", _cnc_y.ToString());

                label34.Text = sss;

                connectPort2Button.Text = "Disconnect";

            }
            else
            {
                timer2.Stop();
                Switch1.ClosePort_();
                Switch1 = null;
                connectPort2Button.Text = "Connect to port";
            }
        }

        private void Send2Button_Click(object sender, EventArgs e)
        {
            string cmd = send2TextBox.Text + (char)13 + (char)10;
            Switch1.SendCmd(cmd);
        }

        private void timer2_Tick(object sender, EventArgs e)
        {
            if (Switch1.port.BytesToRead > 0)
            {
                string txt = Switch1.GetAcceptedKiril();
                resultsTextBox.AppendText(txt + "\n");
            }
        }

        private void LeftButton_Click(object sender, EventArgs e)
        {
            _cnc_x = _cnc_x - float.Parse(dxTextBox.Text);

            string sss = String.Concat("g0 x", _cnc_x.ToString(), " y", _cnc_y.ToString());
            string cmd = sss + (char)13 + (char)10;

            Switch1.SendCmd(cmd);
            sss = String.Concat("px = ", _cnc_x.ToString(), " py = ", _cnc_y.ToString());
            label34.Text = sss;
        }

        private void RightButton_Click(object sender, EventArgs e)
        {
            _cnc_x = _cnc_x + float.Parse(dxTextBox.Text);

            string sss = String.Concat("g0 x", _cnc_x.ToString(), " y", _cnc_y.ToString());
            string cmd = sss + (char)13 + (char)10;

            Switch1.SendCmd(cmd);
            sss = String.Concat("px = ", _cnc_x.ToString(), " py = ", _cnc_y.ToString());
            label34.Text = sss;
        }

        private void UpButton_Click(object sender, EventArgs e)
        {
            _cnc_y = _cnc_y + float.Parse(dyTextBox.Text);

            string sss = String.Concat("g0 x", _cnc_x.ToString(), " y", _cnc_y.ToString());
            string cmd = sss + (char)13 + (char)10;

            Switch1.SendCmd(cmd);
            sss = String.Concat("px = ", _cnc_x.ToString(), " py = ", _cnc_y.ToString());
            label34.Text = sss;
        }

        private void DownButton_Click(object sender, EventArgs e)
        {
            _cnc_y = _cnc_y - float.Parse(dyTextBox.Text);

            string sss = String.Concat("g0 x", _cnc_x.ToString(), " y", _cnc_y.ToString());
            string cmd = sss + (char)13 + (char)10;

            Switch1.SendCmd(cmd);
            sss = String.Concat("px = ", _cnc_x.ToString(), " py = ", _cnc_y.ToString());
            label34.Text = sss;
        }

        private void CollectData2Button_Click(object sender, EventArgs e)
        {
            float nx, ny, dx, dy, lx, ly, x0, y0, x1, y1, feed_rate;

            uint samples1 = uint.Parse(stepsBeforeTextBox.Text);
            uint samples2 = uint.Parse(stepsAfterTextBox.Text);
            uint count_avg = uint.Parse(averagingsTextBox.Text);

            masA = new long[samples1 + samples2];
            _arrA = new double[samples1 + samples2];
            _all = int.Parse(averagingsTextBox.Text);

            ushort RANGE_ = _input_ranges[comboRangeA.SelectedIndex];

            //REPLACE FLOAT WITH FIXED POINT

            x0 = float.Parse(x0TextBox.Text);
            y0 = float.Parse(y0TextBox.Text);
            x1 = float.Parse(x1TextBox.Text);
            y1 = float.Parse(y1TextBox.Text);

            nx = float.Parse(nxTextBox.Text);
            ny = float.Parse(nyTextBox.Text);
            lx = x1 - x0;
            ly = y1 - y0;
            dx = lx / nx;
            dy = ly / ny;

            _cnc_x = x0;
            _cnc_y = y0;

            feed_rate = float.Parse(fTextBox.Text);

            string cmd;

            // string sss = String.Concat("g01 x", _cnc_x.ToString().Replace(',', '.'), " y", _cnc_y.ToString().Replace(',', '.'), " f", feed_rate.ToString().Replace(',', '.'));


            if (moveAlongYCheckBox.Checked || moveAlongXCheckBox.Checked)
            {

                string sss = "g01";
                if (moveAlongXCheckBox.Checked)
                {
                    sss = String.Concat(sss, " x", _cnc_x.ToString().Replace(',', '.'));

                }
                if (moveAlongYCheckBox.Checked)
                {
                    sss = String.Concat(sss, " y", _cnc_y.ToString().Replace(',', '.'));
                }
                sss = String.Concat(sss, " f", feed_rate.ToString().Replace(',', '.'));


                cmd = sss + (char)13 + (char)10;
                Switch1.SendCmd(cmd);
                label34.Text = cmd;
                Thread.Sleep(250);

                int ix = 0;

                for (_cnc_x = x0; (_cnc_x <= x1) && (ix < nx); _cnc_x += dx, ix++)

                {
                    int iy = 0;

                    for (_cnc_y = y0; (_cnc_y <= y1) && (iy < ny); _cnc_y += dy, iy++)

                    {
                        timer2.Stop();
                        while (true)
                        {
                            Thread.Sleep(250);
                            Switch1.SendCmd("?" + (char)13 + (char)10);
                            Thread.Sleep(250);

                            string txt = Switch1.GetAcceptedKiril();
                            resultsTextBox.AppendText(txt + "\n");

                            if ((txt.IndexOf("Idle") > -1) && (txt.IndexOf("WCO") == -1))
                                break;


                        }
                        timer2.Start();
                        sss = "g01";

                        if (moveAlongXCheckBox.Checked)
                            sss = String.Concat(sss, " x", _cnc_x.ToString().Replace(',', '.'));


                        if (moveAlongYCheckBox.Checked)
                            sss = String.Concat(sss, " y", _cnc_y.ToString().Replace(',', '.'));

                        sss = String.Concat(sss, " f", feed_rate.ToString().Replace(',', '.'));

                        cmd = sss + (char)13 + (char)10;
                        Thread.Sleep(250);
                        Switch1.SendCmd(cmd);
                        label34.Text = cmd;
                        Thread.Sleep(250);
                        timer2.Stop();

                        while (true)
                        {
                            Switch1.SendCmd("?" + (char)13 + (char)10);
                            Thread.Sleep(25);

                            string txt = Switch1.GetAcceptedKiril();
                            resultsTextBox.AppendText(txt + "\n");
                            if ((txt.IndexOf("Idle") > -1) && (txt.IndexOf("WCO") == -1))
                            {
                                break;
                            }

                        }
                        timer2.Start();
                        if (measureAndSaveCheckBox.Checked)
                        {
                            string dir = filePathTextBox.Text;
                            if (dir[dir.Length - 1] != '\\')
                            {
                                dir = String.Concat(dir, "\\");
                            }

                            Directory.CreateDirectory(dir);

                            int step_i = int.Parse(saveEveryTextBox.Text);
                            timer1.Enabled = true;
                            CollectData(samples1, samples2, count_avg, RANGE_, masA, _arrA, _all);
                            Application.DoEvents();


                            if (suppressStartCheckBox.Checked)
                            {
                                SuppressSpikes(_arrA);
                            }
                            bool _reason = suppressStartCheckBox.Checked || applyAutoRunAvgCheckBox.Checked || autoEliminateAvgCheckBox.Checked;
                            if (_reason)
                            {
                                if (suppressStartCheckBox.Checked)
                                {
                                    SuppressSpikes(_arrA);
                                }
                                if (applyAutoRunAvgCheckBox.Checked)
                                {
                                    RunAvg(ref _arrA, int.Parse(runAvgTextBox.Text));
                                };
                                if (autoEliminateAvgCheckBox.Checked)
                                {
                                    NoOffset(_arrA);
                                };
                            }

                            if (useTimeFilterCheckBox.Checked)
                            {
                                if (useTimeFilterTextBox.Text.Length > 0)
                                {
                                    Complex[] filtr = LoadFromFileC(useTimeFilterTextBox.Text);
                                    _arrA = FuncMult(_arrA, _oscilloscope_timestep, -_oscilloscope_timestep * (double)samples1, filtr);
                                }
                            }

                            if (autoEliminateAvgCheckBox.Checked)
                            {
                                NoOffset(_arrA);
                            };

                            if (useFreqFilterCheckBox.Checked)
                            {
                                if (useFreqFilterTextBox.Text.Length > 0)
                                {
                                    Complex[] f = FurieTransf(_arrA, _oscilloscope_timestep, -_oscilloscope_timestep * (double)samples1,
                                         double.Parse(FTBottomTextBox.Text), double.Parse(FTStepTextBox.Text), int.Parse(FTStepsNumberTextBox.Text));
                                    Complex[] filtr = LoadFromFileC(useFreqFilterTextBox.Text);

                                    f = FuncMult(f, double.Parse(FTStepTextBox.Text), double.Parse(FTBottomTextBox.Text), filtr);

                                    Complex[] restored = FurieTransfReverse(f, _oscilloscope_timestep, -_oscilloscope_timestep * (double)samples1, _arrA.Length, double.Parse(FTBottomTextBox.Text), double.Parse(FTStepTextBox.Text));

                                    for (int k1 = 0; k1 < restored.Length; k1++)
                                        _arrA[k1] = restored[k1].Real * 2;//важно делать умножение на 2 так как интеграл по полубесконечному промежутку

                                }
                            }

                            timer1.Enabled = false;
                            sss = String.Concat("px", _cnc_x.ToString(), "py", _cnc_y.ToString(), ".txt");

                            string fulldir = String.Concat(dir, sss);
                            Action a = () => Save2FileAsync(fulldir, _arrA);

                            Task.Run(a);

                            if (_stop_flag)
                            {
                                _save = 0;
                                _stop_flag = false;
                            }

                        }
                    }
                    Thread.Sleep(5);
                }
            }
        }

        private void EliminateAvgButton_Click(object sender, EventArgs e)
        {
            NoOffset(_arrA);
            string path = String.Concat(filePathTextBox.Text, DateTime.Now.ToString().Replace(':', '_'), "TempCapture.txt");
            Save2File(path, _arrA);
        }

        /// <summary>
        /// Функция проверяющая наличие символа в TextBox'е.
        /// Если символ обнаруживается, выводит сообщение и удаляет его.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void NumericTextField_TextChanged(object sender, EventArgs e)
        {
            System.Windows.Forms.TextBox textField = sender as System.Windows.Forms.TextBox;
            StringBuilder stringBuilder = new StringBuilder(textField.Text);
            if (!string.IsNullOrEmpty(textField.Text) && textField != null)
            {
                if (!IsNumeric(textField.Text))
                {
                    MessageBox.Show("Ошибка! Вводите только цифры.");
                    textField.Text = stringBuilder.Remove(textField.Text.Length - 1, 1).ToString(); // Очищаем текстовое поле
                }
            }
        }
 
        private void button1_Click(object sender, EventArgs e)
        {
            string dir = filePathTextBox.Text;
            if (dir[dir.Length - 1] != '\\')
            {
                dir = String.Concat(dir, "\\");
            }

            Directory.CreateDirectory(dir);

            timer1.Enabled = true;

            uint samples1 = uint.Parse(stepsBeforeTextBox.Text);
            uint samples2 = uint.Parse(stepsAfterTextBox.Text);
            uint count_avg = uint.Parse(averagingsTextBox.Text);

            masA = new long[samples1 + samples2];
            _arrA = new double[samples1 + samples2];
            _all = int.Parse(averagingsTextBox.Text);

            ushort RANGE_ = _input_ranges[comboRangeA.SelectedIndex];

            CollectData_V2(samples1, samples2, count_avg, RANGE_, masA, _arrA, _all);
            Application.DoEvents();

            if (suppressStartCheckBox.Checked)
            {
                SuppressSpikes(_arrA);
            }

            bool _reason =
                suppressStartCheckBox.Checked ||
                applyAutoRunAvgCheckBox.Checked ||
                autoEliminateAvgCheckBox.Checked;

            if (_reason)
            {
                if (suppressStartCheckBox.Checked)
                {
                    SuppressSpikes(_arrA);
                }
                if (applyAutoRunAvgCheckBox.Checked)
                {
                    RunAvg(ref _arrA, int.Parse(runAvgTextBox.Text));
                };
                if (autoEliminateAvgCheckBox.Checked)
                {
                    NoOffset(_arrA);
                };

            }

            if (useTimeFilterCheckBox.Checked)
            {
                if (useTimeFilterTextBox.Text.Length > 0)
                {
                    Complex[] filtr = LoadFromFileC(useTimeFilterTextBox.Text);
                    _arrA = FuncMult(_arrA, _oscilloscope_timestep, -_oscilloscope_timestep * double.Parse(stepsBeforeTextBox.Text), filtr);
                }


            }


            if (useFreqFilterCheckBox.Checked)
            {
                if (useFreqFilterTextBox.Text.Length > 0)
                {
                    Complex[] f = FurieTransf(_arrA, _oscilloscope_timestep, -_oscilloscope_timestep * double.Parse(stepsBeforeTextBox.Text),
                         double.Parse(FTBottomTextBox.Text), double.Parse(FTStepTextBox.Text), int.Parse(FTStepsNumberTextBox.Text));
                    Complex[] filtr = LoadFromFileC(useFreqFilterTextBox.Text);
                    f = FuncMult(f, double.Parse(FTStepTextBox.Text), double.Parse(FTBottomTextBox.Text), filtr);
                    Complex[] restored = FurieTransfReverse(f, _oscilloscope_timestep, -_oscilloscope_timestep * double.Parse(stepsBeforeTextBox.Text), _arrA.Length, double.Parse(FTBottomTextBox.Text), double.Parse(FTStepTextBox.Text));
                    for (int k1 = 0; k1 < restored.Length; k1++)
                    {
                        _arrA[k1] = restored[k1].Real * 2;//важно делать умножение на 2 так как интеграл по полубесконечному промежутку
                    }
                }


            }

            if (autoEliminateAvgCheckBox.Checked)
            {
                NoOffset(_arrA);
            };

            timer1.Enabled = false;
            dir = String.Concat(dir, fileNameTextBox.Text);
            Save2File(dir, _arrA);
            if (_stop_flag)
            {
                _save = 0; _stop_flag = false;
            }

        }

        private void button2_Click(object sender, EventArgs e)

        {
            float nx, ny, dx, dy, lx, ly, x0, y0, x1, y1, feed_rate;

            uint samples1 = uint.Parse(stepsBeforeTextBox.Text);
            uint samples2 = uint.Parse(stepsAfterTextBox.Text);
            uint count_avg = uint.Parse(averagingsTextBox.Text);

            masA = new long[samples1 + samples2];
            _arrA = new double[samples1 + samples2];
            _all = int.Parse(averagingsTextBox.Text);

            ushort RANGE_ = _input_ranges[comboRangeA.SelectedIndex];

            //REPLACE FLOAT WITH FIXED POINT

            x0 = float.Parse(x0TextBox.Text);
            y0 = float.Parse(y0TextBox.Text);
            x1 = float.Parse(x1TextBox.Text);
            y1 = float.Parse(y1TextBox.Text);

            nx = float.Parse(nxTextBox.Text);
            ny = float.Parse(nyTextBox.Text);
            lx = x1 - x0;
            ly = y1 - y0;
            dx = lx / nx;
            dy = ly / ny;

            _cnc_x = x0;
            _cnc_y = y0;

            feed_rate = float.Parse(fTextBox.Text);

            string cmd;

            // string sss = String.Concat("g01 x", _cnc_x.ToString().Replace(',', '.'), " y", _cnc_y.ToString().Replace(',', '.'), " f", feed_rate.ToString().Replace(',', '.'));


            if (moveAlongYCheckBox.Checked || moveAlongXCheckBox.Checked)
            {

                string sss = "g01";
                if (moveAlongXCheckBox.Checked)
                {
                    sss = String.Concat(sss, " x", _cnc_x.ToString().Replace(',', '.'));

                }
                if (moveAlongYCheckBox.Checked)
                {
                    sss = String.Concat(sss, " y", _cnc_y.ToString().Replace(',', '.'));
                }
                sss = String.Concat(sss, " f", feed_rate.ToString().Replace(',', '.'));


                cmd = sss + (char)13 + (char)10;
                Switch1.SendCmd(cmd);
                label34.Text = cmd;
                Thread.Sleep(250);

                int ix = 0;

                for (_cnc_x = x0; (_cnc_x <= x1) && (ix < nx); _cnc_x += dx, ix++)

                {
                    int iy = 0;

                    for (_cnc_y = y0; (_cnc_y <= y1) && (iy < ny); _cnc_y += dy, iy++)

                    {
                        timer2.Stop();
                        while (true)
                        {
                            Thread.Sleep(250);
                            Switch1.SendCmd("?" + (char)13 + (char)10);
                            Thread.Sleep(250);

                            string txt = Switch1.GetAcceptedKiril();
                            resultsTextBox.AppendText(txt + "\n");

                            if ((txt.IndexOf("Idle") > -1) && (txt.IndexOf("WCO") == -1))
                                break;


                        }
                        timer2.Start();
                        sss = "g01";

                        if (moveAlongXCheckBox.Checked)
                            sss = String.Concat(sss, " x", _cnc_x.ToString().Replace(',', '.'));


                        if (moveAlongYCheckBox.Checked)
                            sss = String.Concat(sss, " y", _cnc_y.ToString().Replace(',', '.'));

                        sss = String.Concat(sss, " f", feed_rate.ToString().Replace(',', '.'));

                        cmd = sss + (char)13 + (char)10;
                        Thread.Sleep(250);
                        Switch1.SendCmd(cmd);
                        label34.Text = cmd;
                        Thread.Sleep(250);
                        timer2.Stop();

                        while (true)
                        {
                            Switch1.SendCmd("?" + (char)13 + (char)10);
                            Thread.Sleep(25);

                            string txt = Switch1.GetAcceptedKiril();
                            resultsTextBox.AppendText(txt + "\n");
                            if ((txt.IndexOf("Idle") > -1) && (txt.IndexOf("WCO") == -1))
                            {
                                break;
                            }

                        }
                        timer2.Start();
                        if (measureAndSaveCheckBox.Checked)
                        {
                            string dir = filePathTextBox.Text;
                            if (dir[dir.Length - 1] != '\\')
                            {
                                dir = String.Concat(dir, "\\");
                            }

                            Directory.CreateDirectory(dir);

                            int step_i = int.Parse(saveEveryTextBox.Text);
                            timer1.Enabled = true;
                            CollectData(samples1, samples2, count_avg, RANGE_, masA, _arrA, _all);
                            Application.DoEvents();


                            if (suppressStartCheckBox.Checked)
                            {
                                SuppressSpikes(_arrA);
                            }
                            bool _reason = suppressStartCheckBox.Checked || applyAutoRunAvgCheckBox.Checked || autoEliminateAvgCheckBox.Checked;
                            if (_reason)
                            {
                                if (suppressStartCheckBox.Checked)
                                {
                                    SuppressSpikes(_arrA);
                                }
                                if (applyAutoRunAvgCheckBox.Checked)
                                {
                                    RunAvg(ref _arrA, int.Parse(runAvgTextBox.Text));
                                };
                                if (autoEliminateAvgCheckBox.Checked)
                                {
                                    NoOffset(_arrA);
                                };
                            }

                            if (useTimeFilterCheckBox.Checked)
                            {
                                if (useTimeFilterTextBox.Text.Length > 0)
                                {
                                    Complex[] filtr = LoadFromFileC(useTimeFilterTextBox.Text);
                                    _arrA = FuncMult(_arrA, _oscilloscope_timestep, -_oscilloscope_timestep * (double)samples1, filtr);
                                }
                            }

                            if (autoEliminateAvgCheckBox.Checked)
                            {
                                NoOffset(_arrA);
                            };

                            if (useFreqFilterCheckBox.Checked)
                            {
                                if (useFreqFilterTextBox.Text.Length > 0)
                                {
                                    Complex[] f = FurieTransf(_arrA, _oscilloscope_timestep, -_oscilloscope_timestep * (double)samples1,
                                         double.Parse(FTBottomTextBox.Text), double.Parse(FTStepTextBox.Text), int.Parse(FTStepsNumberTextBox.Text));
                                    Complex[] filtr = LoadFromFileC(useFreqFilterTextBox.Text);

                                    f = FuncMult(f, double.Parse(FTStepTextBox.Text), double.Parse(FTBottomTextBox.Text), filtr);

                                    Complex[] restored = FurieTransfReverse(f, _oscilloscope_timestep, -_oscilloscope_timestep * (double)samples1, _arrA.Length, double.Parse(FTBottomTextBox.Text), double.Parse(FTStepTextBox.Text));

                                    for (int k1 = 0; k1 < restored.Length; k1++)
                                        _arrA[k1] = restored[k1].Real * 2;//важно делать умножение на 2 так как интеграл по полубесконечному промежутку

                                }
                            }

                            timer1.Enabled = false;
                            sss = String.Concat("px", _cnc_x.ToString(), "py", _cnc_y.ToString(), ".txt");

                            string fulldir = String.Concat(dir, sss);
                            Action a = () => Save2FileAsync(fulldir, _arrA);

                            Task.Run(a);

                            if (_stop_flag)
                            {
                                _save = 0;
                                _stop_flag = false;
                            }

                        }
                    }
                    Thread.Sleep(5);
                }
            }
		}
    

        private void loadMeteringButton_Click(object sender, EventArgs e)
        {
            LoadMetering();
        }

        private void saveMeteringButton_Click(object sender, EventArgs e)
        {
            SaveMetering();
        }

        private void applyFilterButton_Click(object sender, EventArgs e)
        {
            ApplyFilter();
        }

        private void undoLastFilterButton_Click(object sender, EventArgs e)
        {
            UndoLastFilter(); 
        }
	}
}