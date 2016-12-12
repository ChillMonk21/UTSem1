/***************************************************************/
/*                                                             */
/*   LC-3b Simulator                                           */
/*                                                             */
/*   EE 460N                                                   */
/*   The University of Texas at Austin                         */
/*                                                             */
/***************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/***************************************************************/
/*                                                             */
/* Files:  ucode        Microprogram file                      */
/*         isaprogram   LC-3b machine language program file    */
/*                                                             */
/***************************************************************/

/***************************************************************/
/* These are the functions you'll have to write.               */
/***************************************************************/

void eval_micro_sequencer();
void cycle_memory();
void eval_bus_drivers();
void drive_bus();
void latch_datapath_values();

/***************************************************************/
/* A couple of useful definitions.                             */
/***************************************************************/
#define FALSE 0
#define TRUE  1

/***************************************************************/
/* Use this to avoid overflowing 16 bits on the bus.           */
/***************************************************************/
#define Low16bits(x) ((x) & 0xFFFF)
#define Low8bits(x) ((x) & 0xFF)
/***************************************************************/
/* Definition of the control store layout.                     */
/***************************************************************/
#define CONTROL_STORE_ROWS 64
#define INITIAL_STATE_NUMBER 18

/***************************************************************/
/* Definition of bit order in control store word.              */
/***************************************************************/
enum CS_BITS {                                                  
    IRD,
    COND1, COND0,
    J5, J4, J3, J2, J1, J0,
    LD_MAR,
    LD_MDR,
    LD_IR,
    LD_BEN,
    LD_REG,
    LD_CC,
    LD_PC,
    GATE_PC,
    GATE_MDR,
    GATE_ALU,
    GATE_MARMUX,
    GATE_SHF,
    PCMUX1, PCMUX0,
    DRMUX,
    SR1MUX,
    ADDR1MUX,
    ADDR2MUX1, ADDR2MUX0,
    MARMUX,
    ALUK1, ALUK0,
    MIO_EN,
    R_W,
    DATA_SIZE,
    LSHF1,
/* MODIFY: you have to add all your new control signals */
    LD_SSP,
    LD_USP,
    SPMUX1, SPMUX0,
    GATE_SP,
    LD_R6,
    CONTROL_STORE_BITS
} CS_BITS;

/***************************************************************/
/* Functions to get at the control bits.                       */
/***************************************************************/
int GetIRD(int *x)           { return(x[IRD]); }
int GetCOND(int *x)          { return((x[COND1] << 1) + x[COND0]); }
int GetJ(int *x)             { return((x[J5] << 5) + (x[J4] << 4) +
				      (x[J3] << 3) + (x[J2] << 2) +
				      (x[J1] << 1) + x[J0]); }
int GetLD_MAR(int *x)        { return(x[LD_MAR]); }
int GetLD_MDR(int *x)        { return(x[LD_MDR]); }
int GetLD_IR(int *x)         { return(x[LD_IR]); }
int GetLD_BEN(int *x)        { return(x[LD_BEN]); }
int GetLD_REG(int *x)        { return(x[LD_REG]); }
int GetLD_CC(int *x)         { return(x[LD_CC]); }
int GetLD_PC(int *x)         { return(x[LD_PC]); }
int GetGATE_PC(int *x)       { return(x[GATE_PC]); }
int GetGATE_MDR(int *x)      { return(x[GATE_MDR]); }
int GetGATE_ALU(int *x)      { return(x[GATE_ALU]); }
int GetGATE_MARMUX(int *x)   { return(x[GATE_MARMUX]); }
int GetGATE_SHF(int *x)      { return(x[GATE_SHF]); }
int GetPCMUX(int *x)         { return((x[PCMUX1] << 1) + x[PCMUX0]); }
int GetDRMUX(int *x)         { return(x[DRMUX]); }
int GetSR1MUX(int *x)        { return(x[SR1MUX]); }
int GetADDR1MUX(int *x)      { return(x[ADDR1MUX]); }
int GetADDR2MUX(int *x)      { return((x[ADDR2MUX1] << 1) + x[ADDR2MUX0]); }
int GetMARMUX(int *x)        { return(x[MARMUX]); }
int GetALUK(int *x)          { return((x[ALUK1] << 1) + x[ALUK0]); }
int GetMIO_EN(int *x)        { return(x[MIO_EN]); }
int GetR_W(int *x)           { return(x[R_W]); }
int GetDATA_SIZE(int *x)     { return(x[DATA_SIZE]); } 
int GetLSHF1(int *x)         { return(x[LSHF1]); }
/* MODIFY: you can add more Get functions for your new control signals */

/***************************************************************/
/* The control store rom.                                      */
/***************************************************************/
int CONTROL_STORE[CONTROL_STORE_ROWS][CONTROL_STORE_BITS];

/***************************************************************/
/* Main memory.                                                */
/***************************************************************/
/* MEMORY[A][0] stores the least significant byte of word at word address A
   MEMORY[A][1] stores the most significant byte of word at word address A 
   There are two write enable signals, one for each byte. WE0 is used for 
   the least significant byte of a word. WE1 is used for the most significant 
   byte of a word. */

#define WORDS_IN_MEM    0x08000 
#define MEM_CYCLES      5
int MEMORY[WORDS_IN_MEM][2];

/***************************************************************/

/***************************************************************/

/***************************************************************/
/* LC-3b State info.                                           */
/***************************************************************/
#define LC_3b_REGS 8

int RUN_BIT;	/* run bit */
int BUS;	/* value of the bus */

typedef struct System_Latches_Struct{

int PC,		/* program counter */
    MDR,	/* memory data register */
    MAR,	/* memory address register */
    IR,		/* instruction register */
    N,		/* n condition bit */
    Z,		/* z condition bit */
    P,		/* p condition bit */
    BEN;        /* ben register */

int READY;	/* ready bit */
  /* The ready bit is also latched as you dont want the memory system to assert it 
     at a bad point in the cycle*/

int REGS[LC_3b_REGS]; /* register file. */

int MICROINSTRUCTION[CONTROL_STORE_BITS]; /* The microintruction */

int STATE_NUMBER; /* Current State Number - Provided for debugging */ 

/* For lab 4 */
int INTV; /* Interrupt vector register */
int EXCV; /* Exception vector register */
int SSP; /* Initial value of system stack pointer */
/* MODIFY: You may add system latches that are required by your implementation */
int USP;
} System_Latches;

/* Data Structure for Latch */

System_Latches CURRENT_LATCHES, NEXT_LATCHES;

/***************************************************************/
/* A cycle counter.                                            */
/***************************************************************/
int CYCLE_COUNT;

/***************************************************************/
/*                                                             */
/* Procedure : help                                            */
/*                                                             */
/* Purpose   : Print out a list of commands.                   */
/*                                                             */
/***************************************************************/
void help() {                                                    
    printf("----------------LC-3bSIM Help-------------------------\n");
    printf("go               -  run program to completion       \n");
    printf("run n            -  execute program for n cycles    \n");
    printf("mdump low high   -  dump memory from low to high    \n");
    printf("rdump            -  dump the register & bus values  \n");
    printf("?                -  display this help menu          \n");
    printf("quit             -  exit the program                \n\n");
}

/***************************************************************/
/*                                                             */
/* Procedure : cycle                                           */
/*                                                             */
/* Purpose   : Execute a cycle                                 */
/*                                                             */
/***************************************************************/
void cycle() {                                                

  eval_micro_sequencer();   
  cycle_memory();
  eval_bus_drivers();
  drive_bus();
  latch_datapath_values();

  CURRENT_LATCHES = NEXT_LATCHES;

  CYCLE_COUNT++;
}

/***************************************************************/
/*                                                             */
/* Procedure : run n                                           */
/*                                                             */
/* Purpose   : Simulate the LC-3b for n cycles.                 */
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
/* Purpose   : Simulate the LC-3b until HALTed.                 */
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

    printf("\nMemory content [0x%0.4x..0x%0.4x] :\n", start, stop);
    printf("-------------------------------------\n");
    for (address = (start >> 1); address <= (stop >> 1); address++)
	printf("  0x%0.4x (%d) : 0x%0.2x%0.2x\n", address << 1, address << 1, MEMORY[address][1], MEMORY[address][0]);
    printf("\n");

    /* dump the memory contents into the dumpsim file */
    fprintf(dumpsim_file, "\nMemory content [0x%0.4x..0x%0.4x] :\n", start, stop);
    fprintf(dumpsim_file, "-------------------------------------\n");
    for (address = (start >> 1); address <= (stop >> 1); address++)
	fprintf(dumpsim_file, " 0x%0.4x (%d) : 0x%0.2x%0.2x\n", address << 1, address << 1, MEMORY[address][1], MEMORY[address][0]);
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
    printf("Cycle Count  : %d\n", CYCLE_COUNT);
    printf("PC           : 0x%0.4x\n", CURRENT_LATCHES.PC);
    printf("IR           : 0x%0.4x\n", CURRENT_LATCHES.IR);
    printf("STATE_NUMBER : 0x%0.4x\n\n", CURRENT_LATCHES.STATE_NUMBER);
    printf("BUS          : 0x%0.4x\n", BUS);
    printf("MDR          : 0x%0.4x\n", CURRENT_LATCHES.MDR);
    printf("MAR          : 0x%0.4x\n", CURRENT_LATCHES.MAR);
    printf("CCs: N = %d  Z = %d  P = %d\n", CURRENT_LATCHES.N, CURRENT_LATCHES.Z, CURRENT_LATCHES.P);
    printf("Registers:\n");
    for (k = 0; k < LC_3b_REGS; k++)
	printf("%d: 0x%0.4x\n", k, CURRENT_LATCHES.REGS[k]);
    printf("\n");

    /* dump the state information into the dumpsim file */
    fprintf(dumpsim_file, "\nCurrent register/bus values :\n");
    fprintf(dumpsim_file, "-------------------------------------\n");
    fprintf(dumpsim_file, "Cycle Count  : %d\n", CYCLE_COUNT);
    fprintf(dumpsim_file, "PC           : 0x%0.4x\n", CURRENT_LATCHES.PC);
    fprintf(dumpsim_file, "IR           : 0x%0.4x\n", CURRENT_LATCHES.IR);
    fprintf(dumpsim_file, "STATE_NUMBER : 0x%0.4x\n\n", CURRENT_LATCHES.STATE_NUMBER);
    fprintf(dumpsim_file, "BUS          : 0x%0.4x\n", BUS);
    fprintf(dumpsim_file, "MDR          : 0x%0.4x\n", CURRENT_LATCHES.MDR);
    fprintf(dumpsim_file, "MAR          : 0x%0.4x\n", CURRENT_LATCHES.MAR);
    fprintf(dumpsim_file, "CCs: N = %d  Z = %d  P = %d\n", CURRENT_LATCHES.N, CURRENT_LATCHES.Z, CURRENT_LATCHES.P);
    fprintf(dumpsim_file, "Registers:\n");
    for (k = 0; k < LC_3b_REGS; k++)
	fprintf(dumpsim_file, "%d: 0x%0.4x\n", k, CURRENT_LATCHES.REGS[k]);
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
/* Procedure : init_control_store                              */
/*                                                             */
/* Purpose   : Load microprogram into control store ROM        */ 
/*                                                             */
/***************************************************************/
void init_control_store(char *ucode_filename) {                 
    FILE *ucode;
    int i, j, index;
    char line[200];

    printf("Loading Control Store from file: %s\n", ucode_filename);

    /* Open the micro-code file. */
    if ((ucode = fopen(ucode_filename, "r")) == NULL) {
	printf("Error: Can't open micro-code file %s\n", ucode_filename);
	exit(-1);
    }

    /* Read a line for each row in the control store. */
    for(i = 0; i < CONTROL_STORE_ROWS; i++) {
	if (fscanf(ucode, "%[^\n]\n", line) == EOF) {
	    printf("Error: Too few lines (%d) in micro-code file: %s\n",
		   i, ucode_filename);
	    exit(-1);
	}

	/* Put in bits one at a time. */
	index = 0;

	for (j = 0; j < CONTROL_STORE_BITS; j++) {
	    /* Needs to find enough bits in line. */
	    if (line[index] == '\0') {
		printf("Error: Too few control bits in micro-code file: %s\nLine: %d\n",
		       ucode_filename, i);
		exit(-1);
	    }
	    if (line[index] != '0' && line[index] != '1') {
		printf("Error: Unknown value in micro-code file: %s\nLine: %d, Bit: %d\n",
		       ucode_filename, i, j);
		exit(-1);
	    }

	    /* Set the bit in the Control Store. */
	    CONTROL_STORE[i][j] = (line[index] == '0') ? 0:1;
	    index++;
	}

	/* Warn about extra bits in line. */
	if (line[index] != '\0')
	    printf("Warning: Extra bit(s) in control store file %s. Line: %d\n",
		   ucode_filename, i);
    }
    printf("\n");
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

/***************************************************************/
/*                                                             */
/* Procedure : initialize                                      */
/*                                                             */
/* Purpose   : Load microprogram and machine language program  */ 
/*             and set up initial state of the machine.        */
/*                                                             */
/***************************************************************/
void initialize(char *ucode_filename, char *program_filename, int num_prog_files) { 
    int i;
    init_control_store(ucode_filename);

    init_memory();
    for ( i = 0; i < num_prog_files; i++ ) {
	load_program(program_filename);
	while(*program_filename++ != '\0');
    }
    CURRENT_LATCHES.Z = 1;
    CURRENT_LATCHES.STATE_NUMBER = INITIAL_STATE_NUMBER;
    memcpy(CURRENT_LATCHES.MICROINSTRUCTION, CONTROL_STORE[INITIAL_STATE_NUMBER], sizeof(int)*CONTROL_STORE_BITS);
    CURRENT_LATCHES.SSP = 0x3000; /* Initial value of system stack pointer */

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
    if (argc < 3) {
	printf("Error: usage: %s <micro_code_file> <program_file_1> <program_file_2> ...\n",
	       argv[0]);
	exit(1);
    }

    printf("LC-3b Simulator\n\n");

    initialize(argv[1], argv[2], argc - 2);

    if ( (dumpsim_file = fopen( "dumpsim", "w" )) == NULL ) {
	printf("Error: Can't open dumpsim file\n");
	exit(-1);
    }

    while (1)
	get_command(dumpsim_file);

}

/***************************************************************/
/* Do not modify the above code, except for the places indicated 
   with a "MODIFY:" comment.

   Do not modify the rdump and mdump functions.

   You are allowed to use the following global variables in your
   code. These are defined above.

   CONTROL_STORE
   MEMORY
   BUS

   CURRENT_LATCHES
   NEXT_LATCHES

   You may define your own local/global variables and functions.
   You may use the functions to get at the control bits defined
   above.

   Begin your code here 	  			       */
/***************************************************************/


void eval_micro_sequencer() {

  /* 
   * Evaluate the address of the next state according to the 
   * micro sequencer logic. Latch the next microinstruction.
   */

    int ird             = GetIRD(CURRENT_LATCHES.MICROINSTRUCTION);
    int cond            = GetCOND(CURRENT_LATCHES.MICROINSTRUCTION);
    int nextState       = GetJ(CURRENT_LATCHES.MICROINSTRUCTION);
    int fetchOpCode     = Low16bits(CURRENT_LATCHES.IR >> 12);
    int ir11            = ((CURRENT_LATCHES.IR & 0x0800) >> 11);
    int ben             = CURRENT_LATCHES.BEN;
    int ready           = CURRENT_LATCHES.READY;

    if(ird == 1){
        nextState       = fetchOpCode;
    }
    else if(cond == 3){
        if(ir11 == 1){
            nextState       = nextState | 0x01;
        }
    }
    else if(cond == 2){
        if(ben == 1){
            nextState       = nextState | 0x04;
        }
    }
    else if(cond == 1){
        if(ready == 1){
            nextState       = nextState | 0x02;
        }
    }
    else{
        nextState   = nextState;
    }

    NEXT_LATCHES.STATE_NUMBER       = nextState;

    memcpy(NEXT_LATCHES.MICROINSTRUCTION, CONTROL_STORE[NEXT_LATCHES.STATE_NUMBER], sizeof(int)*CONTROL_STORE_BITS);
}

int memCycleCounter;

void cycle_memory() {
 
  /* 
   * This function emulates memory and the WE logic. 
   * Keep track of which cycle of MEMEN we are dealing with.  
   * If fourth, we need to latch Ready bit at the end of 
   * cycle to prepare microsequencer for the fifth cycle.  
   */

    int mioen       = GetMIO_EN(CURRENT_LATCHES.MICROINSTRUCTION);
    int dataSize    = GetDATA_SIZE(CURRENT_LATCHES.MICROINSTRUCTION);
    int rw          = GetR_W(CURRENT_LATCHES.MICROINSTRUCTION);
    int mar         = CURRENT_LATCHES.MAR;
    int mdr         = CURRENT_LATCHES.MDR;
    int memBit      = (mar & 0x1);
    int memLoc      = (mar >> 1);
    int highMDR     = (Low16bits(mdr) >> 8);
    int lowMDR      = Low16bits(mdr & 0xFF);


    if(mioen == 0){
        memCycleCounter         = 0;
        NEXT_LATCHES.READY      = 0;
    }
    else{
        memCycleCounter++;

        if(memCycleCounter >= 4){
            NEXT_LATCHES.READY      = 1;
            memCycleCounter         = 0;
        }
    }

    if((CURRENT_LATCHES.READY == 1) & (mioen == 1)){
        if(rw == 1){
            if(dataSize == 1){
                MEMORY[memLoc][0]       = lowMDR;
                MEMORY[memLoc][1]       = highMDR;
            }
            else{
                MEMORY[memLoc][memBit]  = lowMDR;
            }

            memCycleCounter         = 0;
            NEXT_LATCHES.READY      = 0;
        }
    }

}

int marMuxOutput, pcOutput, shfOutput, aluOutput, mdrOutput, spMuxOutput; 

void eval_bus_drivers() {

  /* 
   * Datapath routine emulating operations before driving the bus.
   * Evaluate the input of tristate drivers 
   *             Gate_MARMUX,
   *         Gate_PC,
   *         Gate_ALU,
   *         Gate_SHF,
   *         Gate_MDR.
   */    
    int marMux          = GetMARMUX(CURRENT_LATCHES.MICROINSTRUCTION);
    int addr1Mux        = GetADDR1MUX(CURRENT_LATCHES.MICROINSTRUCTION);
    int lshf1           = GetLSHF1(CURRENT_LATCHES.MICROINSTRUCTION);
    int addr2Mux        = GetADDR2MUX(CURRENT_LATCHES.MICROINSTRUCTION);
    int sr1Mux          = GetSR1MUX(CURRENT_LATCHES.MICROINSTRUCTION);
    int aluk            = GetALUK(CURRENT_LATCHES.MICROINSTRUCTION);
    int dataSize        = GetDATA_SIZE(CURRENT_LATCHES.MICROINSTRUCTION);
    int spMuxValue	= GetSPMux(CURRENT_LATCHES.MICROINSTRUCTION);

    int addr1MuxOutput;
    int lshf1Output;
    int addr2MuxOutput;
    int offset6;
    int pcOffset9;
    int pcOffset11;
    int addrOutput;
    int baseReg;
    int ldReg6 		= GetLD_REG6(CURRENT.LATCHES.MICROINSTRUCTION);
    int ldReg           = GetLD_REG(CURRENT_LATCHES.MICROINSTRUCTION);
    int drMux           = GetDRMUX(CURRENT_LATCHES.MICROINSTRUCTION);
    int ldMAR           = GetLD_MAR(CURRENT_LATCHES.MICROINSTRUCTION);
    int ldMDR           = GetLD_MDR(CURRENT_LATCHES.MICROINSTRUCTION);
    int ldIR            = GetLD_IR(CURRENT_LATCHES.MICROINSTRUCTION);
    int ldBEN           = GetLD_BEN(CURRENT_LATCHES.MICROINSTRUCTION);
    int ldCC            = GetLD_CC(CURRENT_LATCHES.MICROINSTRUCTION);
    int ldPC            = GetLD_PC(CURRENT_LATCHES.MICROINSTRUCTION);
    int mioMux          = GetMIO_EN(CURRENT_LATCHES.MICROINSTRUCTION);
    int dataSize        = GetDATA_SIZE(CURRENT_LATCHES.MICROINSTRUCTION);

    int irReg           = Low16bits(CURRENT_LATCHES.IR);
    int mar             = Low16bits(CURRENT_LATCHES.MAR);
    int ready           = CURRENT_LATCHES.READY;

    int addr1MuxOutput;
    int lshf1Output;
    int addr2MuxOutput;
    int offset6;
    int pcOffset9;
    int pcOffset11;
    int addrOutput;
    int baseReg;
    int destReg;
    int pcMuxOutput;
    int mioMuxOutput;
    int memLoc;
    int memBit;
    int busSignBit; 

    /*PC*/
    if(pcMux == 0){
        pcMuxOutput           = Low16bits(CURRENT_LATCHES.PC + 2);
    }
    else if(pcMux == 1){
        pcMuxOutput           = Low16bits(BUS);
    }
    else if(pcMux == 2){
        if(addr1Mux == 1){
            baseReg             = Low16bits((CURRENT_LATCHES.IR >> 6) & 0x7);
            addr1MuxOutput      = Low16bits(CURRENT_LATCHES.REGS[baseReg]);
        }
        else{
            addr1MuxOutput      = Low16bits(CURRENT_LATCHES.PC);
        }

        if(addr2Mux == 0){
            addr2MuxOutput      = 0;
        }
        else if(addr2Mux == 1){
            offset6             = Low16bits(CURRENT_LATCHES.IR & 0x3F);
            addr2MuxOutput      = Low16bits(offset6 | ((offset6 >> 5) ? 0xFFC0 : 0)); 

        }
        else if(addr2Mux == 2){
            pcOffset9           = Low16bits(CURRENT_LATCHES.IR & 0x1FF);
            addr2MuxOutput      = Low16bits(pcOffset9 | ((pcOffset9 >> 8) ? 0xFE00 : 0));
        }
        else{
            pcOffset11          = Low16bits(CURRENT_LATCHES.IR & 0x7FF);
            addr2MuxOutput      = Low16bits(pcOffset11 | ((pcOffset11 >> 10) ? 0xF800 : 0));
        }

        if(lshf1 == 0){
            lshf1Output         = Low16bits(addr2MuxOutput);
        }
        else{
            lshf1Output         = Low16bits(addr2MuxOutput << 1);
        }

        addrOutput              = Low16bits(lshf1Output + addr1MuxOutput);

        pcMuxOutput             = Low16bits(addrOutput);

    }

    if(ldPC){
        NEXT_LATCHES.PC         = Low16bits(pcMuxOutput);
    }


    /*LOAD DESTINATION REG*/
    if(drMux == 1){
        destReg             = 7;
    }
    else{
        destReg             = ((irReg >> 9) & 0x7);
    }

    if(ldReg){
        NEXT_LATCHES.REGS[destReg]      = Low16bits(BUS);
    }

    /*LOAD MAR*/
    if(ldMAR){
        NEXT_LATCHES.MAR                = Low16bits(BUS);
    }

    /*LOAD MDR*/
    memLoc            = (mar >> 1);
    memBit            = (mar & 0x1);

    if(mioMux == 0){
        if(dataSize == 0){  
            mioMuxOutput            = Low16bits((BUS & 0xFF) | ((BUS & 0xFF) << 8));
        }
        else{
            mioMuxOutput            = Low16bits(BUS);
        }
    }
    else{
        if(ready == 1){
            mioMuxOutput            = Low8bits(MEMORY[memLoc][0]) + (Low8bits(MEMORY[memLoc][1]) << 8);
            NEXT_LATCHES.READY      = 0;
        }
    }

    if(ldMDR){
        NEXT_LATCHES.MDR            = Low16bits(mioMuxOutput);
    }

    /*LOAD IR
    printf("LOAD IR %d\n", ldIR);   */
    if(ldIR){
        NEXT_LATCHES.IR             = Low16bits(BUS);
    }

    /*LOAD BEN*/
    if(ldBEN){
        NEXT_LATCHES.BEN            =    (CURRENT_LATCHES.N & ((irReg >> 11) & 0x1))
                                       | (CURRENT_LATCHES.Z & ((irReg >> 10) & 0x1))
                                       | (CURRENT_LATCHES.P & ((irReg >> 9) & 0x1));
    }

    busSignBit		= (BUS >> 15);
    /*LOAD CC*/
    if(ldCC){
        if(BUS == 0){
            NEXT_LATCHES.N          = 0;
            NEXT_LATCHES.Z          = 1;
            NEXT_LATCHES.P          = 0;
        }
        else if(busSignBit == 0){
            NEXT_LATCHES.N          = 0;
            NEXT_LATCHES.Z          = 0;
            NEXT_LATCHES.P          = 1;
        }
        else if(busSignBit == 1){
            NEXT_LATCHES.N          = 1;
            NEXT_LATCHES.Z          = 0;
            NEXT_LATCHES.P          = 0;
        }
    }
	
    /* LOAD Register R6*/
    if(ldReg6){
	NEXT.LATCHES.REGS[6]		= Low16bits(BUS);
    }	

 /*   printf("===STATE_VALUES==================================================================================\n");
    printf("\tBUS\t\t:\t%x\n", BUS);
    int i = 0;
    printf("===CURRENT_LATCHES===============================================================================\n");
    printf("\tPC\t\t:\t%x\n", CURRENT_LATCHES.PC);
    printf("\tMDR\t\t:\t%x\n", CURRENT_LATCHES.MDR);
    printf("\tMAR\t\t:\t%x\n", CURRENT_LATCHES.MAR);
    printf("\tIR\t\t:\t%x\n", CURRENT_LATCHES.IR);
    printf("\tN\t\t:\t%x\n", CURRENT_LATCHES.N);
    printf("\tZ\t\t:\t%x\n", CURRENT_LATCHES.Z);
    printf("\tP\t\t:\t%x\n", CURRENT_LATCHES.P);
    printf("\tBEN\t\t:\t%x\n", CURRENT_LATCHES.BEN);
    printf("\tREADY\t\t:\t%x\n", CURRENT_LATCHES.READY);
    printf("\tSTATE_NUMBER\t:\t%d\n", CURRENT_LATCHES.STATE_NUMBER);
    for(i =0; i <8; i++){
        printf("\tREGS[%d]\t\t:\t%x\n", i, CURRENT_LATCHES.REGS[i]);
    }
    printf("\tIRD\t\t:\t%x\n", GetIRD(CURRENT_LATCHES.MICROINSTRUCTION));
    printf("\tCOND\t\t:\t%x\n", GetCOND(CURRENT_LATCHES.MICROINSTRUCTION));
    printf("\tJ\t\t:\t%d\n", GetJ(CURRENT_LATCHES.MICROINSTRUCTION));

    printf("===NEXT_LATCHES=================================================================================\n");
    printf("\tPC\t\t:\t%x\n", NEXT_LATCHES.PC);
    printf("\tMDR\t\t:\t%x\n", NEXT_LATCHES.MDR);
    printf("\tMAR\t\t:\t%x\n", NEXT_LATCHES.MAR);
    printf("\tIR\t\t:\t%x\n", NEXT_LATCHES.IR);
    printf("\tN\t\t:\t%x\n", NEXT_LATCHES.N);
    printf("\tZ\t\t:\t%x\n", NEXT_LATCHES.Z);
    printf("\tP\t\t:\t%x\n", NEXT_LATCHES.P);
    printf("\tBEN\t\t:\t%x\n", NEXT_LATCHES.BEN);
    printf("\tREADY\t\t:\t%x\n", NEXT_LATCHES.READY);
    printf("\tSTATE_NUMBER\t:\t%d\n", NEXT_LATCHES.STATE_NUMBER);
    for(i =0; i <8; i++){
        printf("\tREGS[%d]\t\t:\t%x\n", i, NEXT_LATCHES.REGS[i]);
    }
    printf("\tIRD\t\t:\t%x\n", GetIRD(NEXT_LATCHES.MICROINSTRUCTION));
    printf("\tCOND\t\t:\t%x\n", GetCOND(NEXT_LATCHES.MICROINSTRUCTION));
    printf("\tJ\t\t:\t%x\n", GetJ(NEXT_LATCHES.MICROINSTRUCTION));*/


}
