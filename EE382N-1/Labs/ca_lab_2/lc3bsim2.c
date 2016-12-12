/*
    Name 1: Jayant Bedwal 
    Name 2: Kamayani Rai
    UTEID 1: jb68546
    UTEID 2: kr28735
*/

/***************************************************************/
/*                                                             */
/*   LC-3b Instruction Level Simulator                         */
/*                                                             */
/*   EE 460N                                                   */
/*   The University of Texas at Austin                         */
/*                                                             */
/***************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/***************************************************************/
/*                                                             */
/* Files: isaprogram   LC-3b machine language program file     */
/*                                                             */
/***************************************************************/

/***************************************************************/
/* These are the functions you'll have to write.               */
/***************************************************************/

void process_instruction();

/***************************************************************/
/* A couple of useful definitions.                             */
/***************************************************************/
#define FALSE 0
#define TRUE  1

/***************************************************************/
/* Use this to avoid overflowing 16 bits on the bus.           */
/***************************************************************/
#define Low16bits(x) ((x) & 0xFFFF)

/***************************************************************/
/* Main memory.                                                */
/***************************************************************/
/* MEMORY[A][0] stores the least significant byte of word at word address A
   MEMORY[A][1] stores the most significant byte of word at word address A 
*/

#define WORDS_IN_MEM    0x08000 
int MEMORY[WORDS_IN_MEM][2];

/***************************************************************/

/***************************************************************/

/***************************************************************/
/* LC-3b State info.                                           */
/***************************************************************/
#define LC_3b_REGS 8

int RUN_BIT;	/* run bit */


typedef struct System_Latches_Struct{

  int PC,		/* program counter */
    N,		/* n condition bit */
    Z,		/* z condition bit */
    P;		/* p condition bit */
  int REGS[LC_3b_REGS]; /* register file. */
} System_Latches;

/* Data Structure for Latch */

System_Latches CURRENT_LATCHES, NEXT_LATCHES;

/***************************************************************/
/* A cycle counter.                                            */
/***************************************************************/
int INSTRUCTION_COUNT;

/***************************************************************/
/*                                                             */
/* Procedure : help                                            */
/*                                                             */
/* Purpose   : Print out a list of commands                    */
/*                                                             */
/***************************************************************/
void help() {                                                    
  printf("----------------LC-3b ISIM Help-----------------------\n");
  printf("go               -  run program to completion         \n");
  printf("run n            -  execute program for n instructions\n");
  printf("mdump low high   -  dump memory from low to high      \n");
  printf("rdump            -  dump the register & bus values    \n");
  printf("?                -  display this help menu            \n");
  printf("quit             -  exit the program                  \n\n");
}

/***************************************************************/
/*                                                             */
/* Procedure : cycle                                           */
/*                                                             */
/* Purpose   : Execute a cycle                                 */
/*                                                             */
/***************************************************************/
void cycle() {                                                

  process_instruction();
  CURRENT_LATCHES = NEXT_LATCHES;
  INSTRUCTION_COUNT++;
}

/***************************************************************/
/*                                                             */
/* Procedure : run n                                           */
/*                                                             */
/* Purpose   : Simulate the LC-3b for n cycles                 */
/*                                                             */
/***************************************************************/
void run(int num_cycles) {                                      
  int i;

  if (RUN_BIT == FALSE) {
    printf("Can't simulate, Simulator is halted\n\n");
    return;
  }

  printf("Simulating for %d cycles...\n\n", num_cycles);
  for (i = 0; i < num_cycles; i++) {
    if (CURRENT_LATCHES.PC == 0x0000) {
	    RUN_BIT = FALSE;
	    printf("Simulator halted\n\n");
	    break;
    }
    cycle();
  }
}

/***************************************************************/
/*                                                             */
/* Procedure : go                                              */
/*                                                             */
/* Purpose   : Simulate the LC-3b until HALTed                 */
/*                                                             */
/***************************************************************/
void go() {                                                     
  if (RUN_BIT == FALSE) {
    printf("Can't simulate, Simulator is halted\n\n");
    return;
  }

  printf("Simulating...\n\n");
  while (CURRENT_LATCHES.PC != 0x0000)
    cycle();
  RUN_BIT = FALSE;
  printf("Simulator halted\n\n");
}

/***************************************************************/ 
/*                                                             */
/* Procedure : mdump                                           */
/*                                                             */
/* Purpose   : Dump a word-aligned region of memory to the     */
/*             output file.                                    */
/*                                                             */
/***************************************************************/
void mdump(FILE * dumpsim_file, int start, int stop) {          
  int address; /* this is a byte address */

  printf("\nMemory content [0x%.4x..0x%.4x] :\n", start, stop);
  printf("-------------------------------------\n");
  for (address = (start >> 1); address <= (stop >> 1); address++)
    printf("  0x%.4x (%d) : 0x%.2x%.2x\n", address << 1, address << 1, MEMORY[address][1], MEMORY[address][0]);
  printf("\n");

  /* dump the memory contents into the dumpsim file */
  fprintf(dumpsim_file, "\nMemory content [0x%.4x..0x%.4x] :\n", start, stop);
  fprintf(dumpsim_file, "-------------------------------------\n");
  for (address = (start >> 1); address <= (stop >> 1); address++)
    fprintf(dumpsim_file, " 0x%.4x (%d) : 0x%.2x%.2x\n", address << 1, address << 1, MEMORY[address][1], MEMORY[address][0]);
  fprintf(dumpsim_file, "\n");
  fflush(dumpsim_file);
}

/***************************************************************/
/*                                                             */
/* Procedure : rdump                                           */
/*                                                             */
/* Purpose   : Dump current register and bus values to the     */   
/*             output file.                                    */
/*                                                             */
/***************************************************************/
void rdump(FILE * dumpsim_file) {                               
  int k; 

  printf("\nCurrent register/bus values :\n");
  printf("-------------------------------------\n");
  printf("Instruction Count : %d\n", INSTRUCTION_COUNT);
  printf("PC                : 0x%.4x\n", CURRENT_LATCHES.PC);
  printf("CCs: N = %d  Z = %d  P = %d\n", CURRENT_LATCHES.N, CURRENT_LATCHES.Z, CURRENT_LATCHES.P);
  printf("Registers:\n");
  for (k = 0; k < LC_3b_REGS; k++)
    printf("%d: 0x%.4x\n", k, CURRENT_LATCHES.REGS[k]);
  printf("\n");

  /* dump the state information into the dumpsim file */
  fprintf(dumpsim_file, "\nCurrent register/bus values :\n");
  fprintf(dumpsim_file, "-------------------------------------\n");
  fprintf(dumpsim_file, "Instruction Count : %d\n", INSTRUCTION_COUNT);
  fprintf(dumpsim_file, "PC                : 0x%.4x\n", CURRENT_LATCHES.PC);
  fprintf(dumpsim_file, "CCs: N = %d  Z = %d  P = %d\n", CURRENT_LATCHES.N, CURRENT_LATCHES.Z, CURRENT_LATCHES.P);
  fprintf(dumpsim_file, "Registers:\n");
  for (k = 0; k < LC_3b_REGS; k++)
    fprintf(dumpsim_file, "%d: 0x%.4x\n", k, CURRENT_LATCHES.REGS[k]);
  fprintf(dumpsim_file, "\n");
  fflush(dumpsim_file);
}

/***************************************************************/
/*                                                             */
/* Procedure : get_command                                     */
/*                                                             */
/* Purpose   : Read a command from standard input.             */  
/*                                                             */
/***************************************************************/
void get_command(FILE * dumpsim_file) {                         
  char buffer[20];
  int start, stop, cycles;

  printf("LC-3b-SIM> ");

  scanf("%s", buffer);
  printf("\n");

  switch(buffer[0]) {
  case 'G':
  case 'g':
    go();
    break;

  case 'M':
  case 'm':
    scanf("%i %i", &start, &stop);
    mdump(dumpsim_file, start, stop);
    break;

  case '?':
    help();
    break;
  case 'Q':
  case 'q':
    printf("Bye.\n");
    exit(0);

  case 'R':
  case 'r':
    if (buffer[1] == 'd' || buffer[1] == 'D')
	    rdump(dumpsim_file);
    else {
	    scanf("%d", &cycles);
	    run(cycles);
    }
    break;

  default:
    printf("Invalid Command\n");
    break;
  }
}

/***************************************************************/
/*                                                             */
/* Procedure : init_memory                                     */
/*                                                             */
/* Purpose   : Zero out the memory array                       */
/*                                                             */
/***************************************************************/
void init_memory() {                                           
  int i;

  for (i=0; i < WORDS_IN_MEM; i++) {
    MEMORY[i][0] = 0;
    MEMORY[i][1] = 0;
  }
}

/**************************************************************/
/*                                                            */
/* Procedure : load_program                                   */
/*                                                            */
/* Purpose   : Load program and service routines into mem.    */
/*                                                            */
/**************************************************************/
void load_program(char *program_filename) {                   
  FILE * prog;
  int ii, word, program_base;

  /* Open program file. */
  prog = fopen(program_filename, "r");
  if (prog == NULL) {
    printf("Error: Can't open program file %s\n", program_filename);
    exit(-1);
  }

  /* Read in the program. */
  if (fscanf(prog, "%x\n", &word) != EOF)
    program_base = word >> 1;
  else {
    printf("Error: Program file is empty\n");
    exit(-1);
  }

  ii = 0;
  while (fscanf(prog, "%x\n", &word) != EOF) {
    /* Make sure it fits. */
    if (program_base + ii >= WORDS_IN_MEM) {
	    printf("Error: Program file %s is too long to fit in memory. %x\n",
             program_filename, ii);
	    exit(-1);
    }

    /* Write the word to memory array. */
    MEMORY[program_base + ii][0] = word & 0x00FF;
    MEMORY[program_base + ii][1] = (word >> 8) & 0x00FF;
    ii++;
  }

  if (CURRENT_LATCHES.PC == 0) CURRENT_LATCHES.PC = (program_base << 1);

  printf("Read %d words from program into memory.\n\n", ii);
}

/************************************************************/
/*                                                          */
/* Procedure : initialize                                   */
/*                                                          */
/* Purpose   : Load machine language program                */ 
/*             and set up initial state of the machine.     */
/*                                                          */
/************************************************************/
void initialize(char *program_filename, int num_prog_files) { 
  int i;

  init_memory();
  for ( i = 0; i < num_prog_files; i++ ) {
    load_program(program_filename);
    while(*program_filename++ != '\0');
  }
  CURRENT_LATCHES.Z = 1;  
  NEXT_LATCHES = CURRENT_LATCHES;
    
  RUN_BIT = TRUE;
}

/***************************************************************/
/*                                                             */
/* Procedure : main                                            */
/*                                                             */
/***************************************************************/
int main(int argc, char *argv[]) {                              
  FILE * dumpsim_file;

  /* Error Checking */
  if (argc < 2) {
    printf("Error: usage: %s <program_file_1> <program_file_2> ...\n",
           argv[0]);
    exit(1);
  }

  printf("LC-3b Simulator\n\n");

  initialize(argv[1], argc - 1);

  if ( (dumpsim_file = fopen( "dumpsim", "w" )) == NULL ) {
    printf("Error: Can't open dumpsim file\n");
    exit(-1);
  }

  while (1)
    get_command(dumpsim_file);
    
}

/***************************************************************/
/* Do not modify the above code.
   You are allowed to use the following global variables in your
   code. These are defined above.

   MEMORY

   CURRENT_LATCHES
   NEXT_LATCHES

   You may define your own local/global variables and functions.
   You may use the functions to get at the control bits defined
   above.

   Begin your code here 	  			       */

/***************************************************************/

#define Low8bits(x) ((x) & 0xFF)

void process_instruction(){
  /*  function: process_instruction
   *  
   *    Process one instruction at a time  
   *       -Fetch one instruction
   *       -Decode 
   *       -Execute
   *       -Update NEXT_LATCHES
   */     
	/** KR*/
	int i = 0, j = 0;
   /* Convert PC Offset to unsigned number or change the logic
    * Check the value of PC for jump, jsrr and branch conditions */

        printf("process_instruction getting executed\n");
        printf("======================CURRENT LATCHES======================\n");
        printf("PC\t=\t%x\n", CURRENT_LATCHES.PC);
        printf("N Flag\t=\t%x\n", CURRENT_LATCHES.N);
        printf("Z Flag\t=\t%x\n", CURRENT_LATCHES.Z);
        printf("P Flag\t=\t%x\n", CURRENT_LATCHES.P);
        for(i = 0; i < LC_3b_REGS; i++){
                printf("REG[%d]\t=\t%x\n", i, CURRENT_LATCHES.REGS[i]);
        }
        printf("\n");

        int prgmCnt             = CURRENT_LATCHES.PC;
        int memLoc              = prgmCnt >> 1;
	/* KR */
        int cycleInstU          = Low8bits(MEMORY[memLoc][1]);
        int cycleInstL          = Low8bits(MEMORY[memLoc][0]);

        int instDecode          = cycleInstU >> 4;
        int S3BitDecode         = (cycleInstU & 0xF) >> 1;
        int pcOffset            = 0;
        int pcOffsetSign        = 0;
        int baseOffset          = 0;
        int baseOffsetSign      = 0;
        int destReg             = 0;
        int srcReg1             = 0;
        int srcReg2             = 0;
        int immVal              = 0;
        int src2Sel             = 0;
        int inst2Sel            = 0;
        int baseReg             = 0;
        int shiftType           = 0;
        int shiftAmt            = 0;

        int tempValue           = 0;
        int tempReg             = 0;
        int tempMemL            = 0;
        int tempMemU            = 0;
	/* KR */
        int trapvector		= 0;
        int trap_routine_address	= 0;

        if(instDecode == 0){
                printf("Branch Instruction\n");

                pcOffsetSign            = (cycleInstU & 0x1);
                pcOffset                = cycleInstL << 1;
                pcOffset                = pcOffset | ((pcOffsetSign) ? 0xFE00 : 0);
		/** KR */
		pcOffset		= Low16bits(pcOffset);

                if(S3BitDecode == 0){

                        NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.PC + pcOffset + 2);
                }
                else if(S3BitDecode == 1){
                        if(CURRENT_LATCHES.P == 1){

                                NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.PC + pcOffset + 2);  
                        }
                        else{
                                NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.PC + 2);
                        }
                }
                else if(S3BitDecode == 2){
                        if(CURRENT_LATCHES.Z == 1){
                                NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.PC + pcOffset + 2); 
                        }
                        else{
                                NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.PC + 2);
                        }
                }
                else if(S3BitDecode == 4){
                        if(CURRENT_LATCHES.N == 1){
                                NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.PC + pcOffset + 2);  
                        }
                        else{
                                NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.PC + 2);
                        }
                }
                else if(S3BitDecode == 3){
                        if((CURRENT_LATCHES.Z == 1) | (CURRENT_LATCHES.P == 1)){
                                NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.PC + pcOffset + 2);  
                        }
                        else{
                                NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.PC + 2);
                        }
                }
                else if(S3BitDecode == 5){
                        if((CURRENT_LATCHES.N == 1) | (CURRENT_LATCHES.P == 1)){
                                NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.PC + pcOffset + 2); 
                        }
                        else{
                                NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.PC + 2);
                        }
                }
                else if(S3BitDecode == 6){
                        if((CURRENT_LATCHES.N == 1) | (CURRENT_LATCHES.Z == 1)){
                                NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.PC + pcOffset + 2);  
                        }
                        else{
                                NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.PC + 2);
                        }
                }
                else if(S3BitDecode == 7){
                        if((CURRENT_LATCHES.N == 1) | (CURRENT_LATCHES.Z == 1) | (CURRENT_LATCHES.P == 1)){
                                NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.PC + pcOffset + 2);
                        }
                        else{
                                NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.PC + 2);
                        }
                }
        }

        else if(instDecode == 1){
                printf("ADD instruction\n");

                destReg         = S3BitDecode;
                srcReg1         = ((cycleInstL & 0xC0) >> 6) + ((cycleInstU & 0x1) << 2);
                src2Sel         = (cycleInstL & 0x20) >> 5;
                srcReg2         = (cycleInstL & 0x7);
                immVal          = (cycleInstL & 0x1F);
                immVal          = immVal | ((immVal & 0x10) ? 0xFFE0 : 0); 

                

                if(src2Sel == 1){
                        tempValue   = Low16bits(CURRENT_LATCHES.REGS[srcReg1] + immVal);      
                }
                else{
                        tempValue   = Low16bits(CURRENT_LATCHES.REGS[srcReg1] + CURRENT_LATCHES.REGS[srcReg2]);
                }

                NEXT_LATCHES.PC                 = Low16bits(CURRENT_LATCHES.PC + 2);
                NEXT_LATCHES.REGS[destReg]      = tempValue;

                if(tempValue == 0){
                        NEXT_LATCHES.N          = 0;
                        NEXT_LATCHES.Z          = 1;
                        NEXT_LATCHES.P          = 0;
                }
                else if((tempValue >> 15) == 0){
                        NEXT_LATCHES.N          = 0;
                        NEXT_LATCHES.Z          = 0;
                        NEXT_LATCHES.P          = 1;
                }
                else{
                        NEXT_LATCHES.N          = 1;
                        NEXT_LATCHES.Z          = 0;
                        NEXT_LATCHES.P          = 0;
                }

                printf("Dest Reg\t=\t%x\n",destReg);
                printf("Src Reg1\t=\t%x\n",srcReg1);
                printf("Src Reg2\t=\t%x\n",srcReg2);
                printf("Src Sel \t=\t%x\n",src2Sel);
                printf("Imm value\t=\t%x\n",immVal);
                printf("Temp Value\t=\t%x\n", tempValue);

                int temp1       = cycleInstL & 0xC0;
                int temp2       = temp1 >> 6;
                int temp3       = cycleInstU & 0x1;
                int temp4       = temp3 << 2;

                printf("cycleInstU = %x\n", cycleInstU);
                printf("cycleInstL = %x\n", cycleInstL);
                printf("temp1      = %x\n", temp1);
                printf("temp2      = %x\n", temp2);
                printf("temp3      = %x\n", temp3);
                printf("temp4      = %x\n", temp4);


        }

        else if(instDecode == 5){
                printf("AND instruction\n");

                destReg         = S3BitDecode;
                srcReg1         = ((cycleInstL & 0xC0) >> 6) + ((cycleInstU & 0x1) << 2);
                src2Sel         = (cycleInstL & 0x20) >> 5;
                srcReg2         = (cycleInstL & 0x7);
                immVal          = (cycleInstL & 0x1F);
                immVal          = immVal | ((immVal & 0x10) ? 0xFFE0 : 0);


                if(src2Sel == 1){
                        tempValue   = Low16bits(CURRENT_LATCHES.REGS[srcReg1] & immVal);      
                }
                else{
                        tempValue   = Low16bits(CURRENT_LATCHES.REGS[srcReg1] & CURRENT_LATCHES.REGS[srcReg2]);
                }

                NEXT_LATCHES.PC                 = Low16bits(CURRENT_LATCHES.PC + 2);
                NEXT_LATCHES.REGS[destReg]      = tempValue;

                if(tempValue == 0){
                        NEXT_LATCHES.N          = 0;
                        NEXT_LATCHES.Z          = 1;
                        NEXT_LATCHES.P          = 0;
                }
                else if((tempValue >> 15) == 0){
                        NEXT_LATCHES.N          = 0;
                        NEXT_LATCHES.Z          = 0;
                        NEXT_LATCHES.P          = 1;
                }
                else{
                        NEXT_LATCHES.N          = 1;
                        NEXT_LATCHES.Z          = 0;
                        NEXT_LATCHES.P          = 0;
                }

                printf("Dest Reg\t=\t%x\n",destReg);
                printf("Src Reg1\t=\t%x\n",srcReg1);
                printf("Src Reg2\t=\t%x\n",srcReg2);
                printf("Src Sel \t=\t%x\n",src2Sel);
                printf("Imm value\t=\t%x\n",immVal);
                printf("Temp Value\t=\t%x\n", tempValue);
        }

        else if(instDecode == 12){
                printf("JMP RET Instruction\n");

                baseReg                 = ((cycleInstL & 0xC0) >> 6) + ((cycleInstU & 0x1) << 2);

                NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.REGS[baseReg]);

        }

        else if(instDecode == 4){
                printf("JSR/JSRR Instruction\n");

                inst2Sel                = ((cycleInstU & 0x8) >> 3);
                tempReg                 = Low16bits(CURRENT_LATCHES.PC + 2);
                baseReg                 = ((cycleInstL & 0xC0) >> 6) + ((cycleInstU & 0x1) << 2);
                pcOffsetSign            = (cycleInstU & 0x4) >> 2;
                pcOffset                = (((cycleInstU & 0x3) << 8) + (cycleInstL)) << 1;              /*JB*/
                pcOffset                = pcOffset | ((pcOffsetSign) ? 0xF800 : 0);
		
                /** KR */
		pcOffset		= Low16bits(pcOffset);

                NEXT_LATCHES.REGS[7]            = Low16bits(tempReg);   /* JB*/

                if(inst2Sel == 0){
                        NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.REGS[baseReg]);
                }
                else{
                        NEXT_LATCHES.PC         = Low16bits(CURRENT_LATCHES.PC + pcOffset + 2);
                }

                

        }

        else if(instDecode == 2){
                printf("LDB Instruction\n");

                destReg                 = S3BitDecode;
                baseReg                 = ((cycleInstL & 0xC0) >> 6) + ((cycleInstU & 0x1) << 2);
                baseOffsetSign          = (cycleInstL & 0x20) >> 5;
                baseOffset              = cycleInstL & 0x1F;
                baseOffset              = baseOffset | ((baseOffsetSign) ? 0xFFE0 : 0);

                memLoc                                  = Low16bits(CURRENT_LATCHES.REGS[baseReg] + baseOffset);
                printf("Mem Location\t=\t%x\n", memLoc);

                if((memLoc & 0x1) == 0){
                        tempValue       = 0;
                }
                else{
                        tempValue       = 1;
                }

                memLoc                                  = memLoc >> 1;

                tempReg                                 = Low16bits(MEMORY[memLoc][tempValue]);
                NEXT_LATCHES.REGS[destReg]              = tempReg;
                NEXT_LATCHES.PC                         = Low16bits(CURRENT_LATCHES.PC + 2);

                if(tempReg == 0){
                        NEXT_LATCHES.N          = 0;
                        NEXT_LATCHES.Z          = 1;
                        NEXT_LATCHES.P          = 0;
                }
                else if((tempReg >> 15) == 0){
                        NEXT_LATCHES.N          = 0;
                        NEXT_LATCHES.Z          = 0;
                        NEXT_LATCHES.P          = 1;
                }
                else{
                        NEXT_LATCHES.N          = 1;
                        NEXT_LATCHES.Z          = 0;
                        NEXT_LATCHES.P          = 0;
                }

                printf("Dest Reg\t=\t%x\n",destReg);
                printf("Base Reg\t=\t%x\n",baseReg);
                printf("Base Offset Sign\t=\t%x\n",baseOffsetSign);
                printf("Base Offset\t=\t%x\n",baseOffset);
                printf("Mem Location\t=\t%x\n", memLoc);
                printf("Temp Value\t=\t%x\n", tempValue);
                printf("Mem Read\t=\t%x\n",tempReg);

        }

        else if(instDecode == 6){
                printf("LDW Instruction\n");

                destReg                 = S3BitDecode;
                baseReg                 = ((cycleInstL & 0xC0) >> 6) + ((cycleInstU & 0x1) << 2);
                baseOffsetSign          = (cycleInstL & 0x20) >> 5;
                baseOffset              = cycleInstL & 0x1F;
                baseOffset              = baseOffset | ((baseOffsetSign) ? 0xFFE0 : 0);

                memLoc                                  = Low16bits((CURRENT_LATCHES.REGS[baseReg] + (baseOffset << 1)) >> 1);

                tempMemL                                = MEMORY[memLoc][0];
                tempMemU                                = MEMORY[memLoc][1];

                tempReg                                 = Low16bits(((tempMemU & 0xFF) << 8) + (tempMemL & 0xFF));
                NEXT_LATCHES.REGS[destReg]              = tempReg;
                NEXT_LATCHES.PC                         = Low16bits(CURRENT_LATCHES.PC + 2);

                if(tempReg == 0){
                        NEXT_LATCHES.N          = 0;
                        NEXT_LATCHES.Z          = 1;
                        NEXT_LATCHES.P          = 0;
                }
                else if((tempReg >> 15) == 0){
                        NEXT_LATCHES.N          = 0;
                        NEXT_LATCHES.Z          = 0;
                        NEXT_LATCHES.P          = 1;
                }
                else{
                        NEXT_LATCHES.N          = 1;
                        NEXT_LATCHES.Z          = 0;
                        NEXT_LATCHES.P          = 0;
                }

                printf("Dest Reg\t=\t%x\n",destReg);
                printf("Base Reg\t=\t%x\n",baseReg);
                printf("Base Offset Sign\t=\t%x\n",baseOffsetSign);
                printf("Base Offset\t=\t%x\n",baseOffset);
                printf("Mem Location\t=\t%x\n", memLoc);
                printf("Mem Read Low\t=\t%x\n",tempMemL);
                printf("Mem Read High\t=\t%x\n",tempMemU);

                int temp1       = baseOffset << 1;
                printf("Temp 1\t=\t%x\n",temp1);


        }

        else if(instDecode == 14){
                printf("LEA Instruction\n");

                destReg                 = S3BitDecode;
                pcOffsetSign            = cycleInstU & 0x1;
                pcOffset                = cycleInstL << 1;
                pcOffset                = pcOffset | ((pcOffsetSign) ? 0xFE00 : 0);
		/** KR */
		pcOffset		= Low16bits(pcOffset);

                NEXT_LATCHES.REGS[destReg]        = Low16bits(CURRENT_LATCHES.PC + pcOffset + 2);

                NEXT_LATCHES.PC                   = Low16bits(CURRENT_LATCHES.PC + 2);

                printf("Dest Reg\t=\t%x\n",destReg);
                printf("PC Offset Sign\t=\t%x\n",pcOffsetSign);
                printf("PC Offset\t=\t%x\n",pcOffset);

                printf("cycleInstU = %x\n", cycleInstU);
                printf("cycleInstL = %x\n", cycleInstL);
        }

        else if(instDecode == 9){
                printf("XOR Instruction\n");

                destReg             = S3BitDecode;
                srcReg1             = ((cycleInstL & 0xC0) >> 6) + ((cycleInstU & 0x1) << 2);
                src2Sel             = (cycleInstL & 0x20) >> 5;
                srcReg2             = (cycleInstL & 0x7);
                immVal              = (cycleInstL & 0x1F);
                immVal              = immVal | ((immVal & 0x10) ? 0xFFE0 : 0);      

                if(src2Sel == 0){
                        tempValue               = Low16bits(CURRENT_LATCHES.REGS[srcReg1] ^ CURRENT_LATCHES.REGS[srcReg2]);
                }
                else{
                        tempValue               = Low16bits(CURRENT_LATCHES.REGS[srcReg1] ^ immVal);
                }
                

                NEXT_LATCHES.REGS[destReg]              = tempValue; 

                NEXT_LATCHES.PC                         = Low16bits(CURRENT_LATCHES.PC + 2);

                if(tempValue == 0){
                        NEXT_LATCHES.N          = 0;
                        NEXT_LATCHES.Z          = 1;
                        NEXT_LATCHES.P          = 0;
                }
                else if((tempValue >> 15) == 0){
                        NEXT_LATCHES.N          = 0;
                        NEXT_LATCHES.Z          = 0;
                        NEXT_LATCHES.P          = 1;
                }
                else{
                        NEXT_LATCHES.N          = 1;
                        NEXT_LATCHES.Z          = 0;
                        NEXT_LATCHES.P          = 0;
                }

                printf("Dest Reg\t=\t%x\n",destReg);
                printf("Src Reg1\t=\t%x\n",srcReg1);
                printf("Src Reg2\t=\t%x\n",srcReg2);
                printf("Src Sel \t=\t%x\n",src2Sel);
                printf("Imm value\t=\t%x\n",immVal);
                printf("Temp Value\t=\t%x\n", tempValue);
        }

        else if(instDecode == 13){
                printf("SHF Instruction\n");

                destReg                 = S3BitDecode;
                srcReg1                 = ((cycleInstL & 0xC0) >> 6) + ((cycleInstU & 0x1) << 2);
                shiftType               = (cycleInstL & 0x30) >> 4;
                shiftAmt                = cycleInstL & 0xF;

                if(shiftType == 0){
                        tempValue               = CURRENT_LATCHES.REGS[srcReg1] << shiftAmt;
                        printf("Temp Value 1\t=\t%x\n", tempValue);
                }
                else if(shiftType == 1){
                        tempValue               = (CURRENT_LATCHES.REGS[srcReg1] >> shiftAmt);
                        printf("Temp Value 1\t=\t%x\n", tempValue);
                }
                else if(shiftType == 3){
                        tempValue               = CURRENT_LATCHES.REGS[srcReg1];
                        printf("Temp Value 1\t=\t%x\n", tempValue);

                        tempValue               = (tempValue) | ((tempValue & 0x8000) ? 0xFFFF0000 : 0);
                        printf("Temp Value 2\t=\t%x\n", tempValue);

                        tempValue               = tempValue >> shiftAmt;
                        printf("Temp Value 3\t=\t%x\n", tempValue);
                }

                tempValue                       = Low16bits(tempValue);
                NEXT_LATCHES.PC                 = Low16bits(CURRENT_LATCHES.PC + 2);

                NEXT_LATCHES.REGS[destReg]      = tempValue;

                if(tempValue == 0){
                        NEXT_LATCHES.N          = 0;
                        NEXT_LATCHES.Z          = 1;
                        NEXT_LATCHES.P          = 0;
                }
                else if((tempValue >> 15) == 0){
                        NEXT_LATCHES.N          = 0;
                        NEXT_LATCHES.Z          = 0;
                        NEXT_LATCHES.P          = 1;
                }
                else{
                        NEXT_LATCHES.N          = 1;
                        NEXT_LATCHES.Z          = 0;
                        NEXT_LATCHES.P          = 0;
                }

                printf("Dest Reg\t=\t%x\n",destReg);
                printf("Src Reg1\t=\t%x\n",srcReg1);
                printf("Temp Value\t=\t%x\n", tempValue);
                printf("Shift Type\t=\t%x\n", shiftType);
                printf("Shift Amount\t=\t%x\n", shiftAmt);
        }

        else if(instDecode == 3){
                 printf("STB Instruction\n");

                destReg                 = S3BitDecode;
                baseReg                 = ((cycleInstL & 0xC0) >> 6) + ((cycleInstU & 0x1) << 2);
                baseOffsetSign          = (cycleInstL & 0x20) >> 5;
                baseOffset              = cycleInstL & 0x1F;
                baseOffset              = baseOffset | ((baseOffsetSign) ? 0xFFE0 : 0);

                memLoc                                  = Low16bits(CURRENT_LATCHES.REGS[baseReg] + baseOffset);
                printf("Mem Location\t=\t%x\n", memLoc);

                if((memLoc & 0x1) == 0){
                        tempValue       = 0;
                }
                else{
                        tempValue       = 1;
                }

                memLoc                                  = memLoc >> 1;

                MEMORY[memLoc][tempValue]               = (CURRENT_LATCHES.REGS[destReg] & 0xFF);

                NEXT_LATCHES.PC                         = Low16bits(CURRENT_LATCHES.PC + 2);

                printf("Dest Reg\t=\t%x\n",destReg);
                printf("Base Reg\t=\t%x\n",baseReg);
                printf("Base Offset Sign\t=\t%x\n",baseOffsetSign);
                printf("Base Offset\t=\t%x\n",baseOffset);
                printf("Mem Location\t=\t%x\n", memLoc);
                printf("Temp Value\t=\t%x\n", tempValue);

        }

        else if(instDecode == 7){
                printf("STW Instruction\n");

                destReg                 = S3BitDecode;
                baseReg                 = ((cycleInstL & 0xcC0) >> 6) + ((cycleInstU & 0x1) << 2);
                baseOffsetSign          = (cycleInstL & 0x20) >> 5;
                baseOffset              = cycleInstL & 0x1F;
                baseOffset              = baseOffset | ((baseOffsetSign) ? 0xFFE0 : 0);

                memLoc                                  = Low16bits((CURRENT_LATCHES.REGS[baseReg] + (baseOffset << 1)) >> 1);

                tempMemL                                = (CURRENT_LATCHES.REGS[destReg] & 0xFF);
                tempMemU                                = (CURRENT_LATCHES.REGS[destReg] >> 8) & 0xFF;

                MEMORY[memLoc][0]                       = tempMemL;
                MEMORY[memLoc][1]                       = tempMemU;

                NEXT_LATCHES.PC                         = Low16bits(CURRENT_LATCHES.PC + 2);


                printf("Dest Reg\t=\t%x\n",destReg);
                printf("Base Reg\t=\t%x\n",baseReg);
                printf("Base Offset Sign\t=\t%x\n",baseOffsetSign);
                printf("Base Offset\t=\t%x\n",baseOffset);
                printf("Mem Location\t=\t%x\n", memLoc);
                printf("Mem Read Low\t=\t%x\n",tempMemL);
                printf("Mem Read High\t=\t%x\n",tempMemU);


        }

        else if(instDecode == 15){
		printf("TRAP Instruction\n");
   		trapvector                              = cycleInstL;

                NEXT_LATCHES.REGS[7]                    = Low16bits(CURRENT_LATCHES.PC + 2);
                NEXT_LATCHES.PC                         = (Low8bits(MEMORY[trapvector][1]) << 8) + Low8bits(MEMORY[trapvector][0]);             

		printf("trapvector = %x, MEMORY[trapvector][1] = %d, MEMORY[trapvector][0] = %d\n",trapvector,MEMORY[trapvector][1],MEMORY[trapvector][0]);
        }


        printf("====================== NEXT  LATCHES ======================\n");
        printf("PC\t=\t%x\n", NEXT_LATCHES.PC);
        printf("N Flag\t=\t%x\n", NEXT_LATCHES.N);
        printf("Z Flag\t=\t%x\n", NEXT_LATCHES.Z);
        printf("P Flag\t=\t%x\n", NEXT_LATCHES.P);
        for(j =0; j < LC_3b_REGS; j++){
                printf("REG[%x]\t=\t%x\n", j, NEXT_LATCHES.REGS[j]);
        }
        printf("\n");

}
