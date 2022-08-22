// Use pin 2 for white LED
const unsigned int white_leds = 0B00000100; // port D pins are D7, D6, D5, D4, D3, D2, RX, TX
const unsigned int initial_on = 0B00000000;

const unsigned long long freq = 1000;
unsigned int counter = 0;
void setup(){
  DDRD = white_leds;
  PORTD = initial_on;
  cli();//stop interrupts
  TCCR1A = 0;// set entire TCCR1A register to 0
  TCCR1B = 0;// same for TCCR1B
  OCR1A = (F_CPU) / (freq*4) - 1; // (must be <65536)
  TCCR1B |= (1 << WGM12); // turn on CTC mode
  TCCR1B |= (1 << CS10); // no prescaler
  TIMSK1 |= (1 << OCIE1A); // enable timer compare interrupt
  sei(); //allow interrupts
}

ISR(TIMER1_COMPA_vect){
  // This runs at 4*freq = 4000 Hz
  // Allows for turning on a secondary LED for finding rolling shutter direction
  if (counter == 0) {
    // turn on white
    PORTD |= white_leds;  
  }
  else if (counter == 1) {

  }
  else if (counter == 2) {
    // turn off white
    PORTD &= ~white_leds;
  }
  else if (counter == 3) {
    
  }
  counter = (counter + 1) % 4;
}
void loop(){}
