int latchPin = 5;
int clockPin = 6;
int dataPin = 7;

int data[] = {0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000, 0b01000000, 0b10000000};
byte bytes[20];
int  first_bytes_read = 0 ; 

void writeData() {
  digitalWrite(latchPin, LOW);
  for (int i = 19; i >=0; i--) {
    shiftOut(dataPin, clockPin, MSBFIRST, bytes[i]);
  }
  digitalWrite(latchPin, HIGH);
}
 

void test() {
  
}
void setup_old () 
{
    pinMode(latchPin, OUTPUT);
  pinMode(dataPin, OUTPUT);
  pinMode(clockPin, OUTPUT);
  Serial.begin( 19200, SERIAL_8E2);
  Serial.setTimeout(3000);
  //test();
}


void setup() {
  pinMode(latchPin, OUTPUT);
  pinMode(dataPin, OUTPUT);
  pinMode(clockPin, OUTPUT);
  Serial.begin( 19200, SERIAL_8E2);
  Serial.setTimeout(3000);

  //trash removal
  while  (Serial.available())
  {
    int ret = Serial.readBytes(bytes, 20); 
  }
    delay(100);
  while  (Serial.available())
  {
    int ret = Serial.readBytes(bytes, 20); 
  }
    delay(100);

  Serial.println ("STARTED"); 
  Serial.flush();


 while  ( Serial.available() == 0 )
  {
    delay(100);
  }

  first_bytes_read = Serial.read();
  Serial.print ("FBR"); 
  Serial.println (first_bytes_read); 
  Serial.flush(); 
}

void accept_code_V1 () 
{

/*
  if (Serial.available()) {

    //String str_in = Serial.readString();

    int ret = Serial.readBytes(bytes, 20);
    //Serial.println(str_in);
    Serial.println(ret);
    //if (ret == 20) {
      writeData();
      Serial.println("ok");
    //}
    
  }
*/
delay(100);

  if (Serial.available()) 
  {

    int ret = Serial.available();

    Serial.print ("P1 "); 
    delay(1);
    Serial.println(ret); 
    Serial.flush();
    delay(1);

    delay(100);
    ret = Serial.available();
 
    Serial.print ("P2 "); 
    Serial.println(ret); 
    Serial.flush();
    delay(1);
    delay(100);


    if  ((Serial.available()<first_bytes_read)||  (first_bytes_read==0))
    {
      delay(500);
      ret = Serial.available();
       
      Serial.print ("P3 "); 
      Serial.println(ret); 
      Serial.flush();
    delay(1);
    }


  int hardcore_check = 10;

  if (hardcore_check)
    while ((Serial.available()!=first_bytes_read) &&(first_bytes_read>0) && hardcore_check){
      hardcore_check--;
      delay(100);
       ret = Serial.available();
       
    Serial.print ("P4 "); 
    Serial.println(ret); 
    Serial.flush(); 
    delay(1);
    }

    if ((Serial.available()>first_bytes_read) &&(first_bytes_read>0)  )
    {
      while (Serial.available() )
      ret = Serial.readBytes(bytes, 20);
    }
    else
    { 
      ret = Serial.readBytes(bytes, 20);
      if (first_bytes_read==0)
        first_bytes_read = ret; 
    }
    
    Serial.flush();
    delay(1);
    Serial.print ("P5 "); 
    Serial.println(ret); 
    Serial.flush();
    delay(1);

    if (ret == first_bytes_read) 
      writeData();

  }
}


void accept_code_V2 () 
{
  delay(100);

  while  (Serial.available()< first_bytes_read) 
    { 
      delay(1);
      Serial.flush();
    }

  delay(500);

  int ret = Serial.readBytes(bytes, 20); 
  if  ( ret==first_bytes_read )
    writeData();

  Serial.println(ret); 
  Serial.flush();
}




 
     //   https://alexgyver.ru/lessons/crc/
    //    CRC (cyclic redundancy code) – циклический избыточный код. Алгоритм тоже выдаёт некое “число” при 
   // прохождении через него потока байтов, но учитывает все предыдущие данные при расчёте. Как работает данный
   //  алгоритм мы рассматривать не будем, об этом можно почитать на Википедии или здесь. Рассмотрим реализацию
   //   CRC 8 бит по стандарту Dallas, он используется в датчиках этой фирмы (например DS18b20 и домофонные ключи 
   //   iButton). Данная реализация должна работать на всех платформах, так как это чисто C++ без привязки к архитектуре
   //    (компилятор сам разберётся):

byte crc8(byte *buffer, byte size) {
  byte crc = 0;
  for (byte i = 0; i < size; i++) 
  {
    byte data = buffer[i];
    for (int j = 8; j > 0; j--) 
    {
      crc = ((crc ^ data) & 1) ? (crc >> 1) ^ 0x8C : (crc >> 1);
      data >>= 1;
    }
  }
  return crc;
}


void accept_code_V3 () 
{
 // delay(100);

  while  (Serial.available()< first_bytes_read) 
    { 
      delay(1);
      Serial.flush();
    }

 // delay(500);

  int ret = Serial.readBytes(bytes, 20); 

  byte  crc_= crc8(bytes,ret  );
  if  ( ret==first_bytes_read )
  { 
    writeData();
  }

  Serial.println(ret); 
  Serial.flush();
  Serial.println((int)crc_); 
  Serial.flush();
}

 
void loop() 
{

//  Serial.print ("FBR"); 
//  Serial.println (first_bytes_read); 
//  Serial.flush(); 
  accept_code_V3 () ;
}
