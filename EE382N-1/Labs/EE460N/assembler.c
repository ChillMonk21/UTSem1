/*
  Name 1: Jayant Bedwal
  Name 2: Kamayani Rai 
  UTEID 1: jb68546
  UTEID 2: kr28735
*/

#include <stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<limits.h>
#include<ctype.h>

#define MAX_LINE_LENGTH 255
#define MAX_LABEL_LENGTH 20
#define MAX_LABEL_COUNT 255
#define MAX_OPCODE_COUNT 28
#define RESTRICTED_LABEL_VALUES_COUNT 8

enum
{
	DONE,
	OK,
	EMPTY_LINE
};

typedef struct
{
	int address;
	char label[MAX_LABEL_LENGTH +1];

}LabelEntry;

int labelCounter 		= 0;
int location_counter	 	= 0;
int first_pass 			= 0;
int firstLine			= 0;
FILE* infile 			= NULL;
FILE* outfile 			= NULL;

LabelEntry labelTable[MAX_LABEL_COUNT];

char valid_opcodes[MAX_OPCODE_COUNT][6] =
{
  	"add", "and", "br", "brn", "brnp", "brnz", "brnzp", "brp", "brz", "brzp", "halt", "jmp", "jsr", "jsrr", "ldb", "ldw", "lea",
  	"nop", "not", "ret", "lshf", "rshfl", "rshfa","rti", "stb", "stw", "trap", "xor"
};

char restricted_label_values [RESTRICTED_LABEL_VALUES_COUNT][6] = 
{
        "in", "out", "getc", "puts", "orig", "blkw", "fill", "string"
};

void process_input_output_files(FILE *fin, FILE *fout);

int readAndParse( FILE *pInfile, char *pLine, char **pLabel, char **pOpcode,
			char ** pArg1, char ** pArg2, char ** pArg3, char ** pArg4);

void check_and_write_output(FILE *fout, char *lLabel, char *lOpcode, char *lArg1,
                       char *lArg2, char *lArg3, char *lArg4);

int check_and_write_pseudo_op(FILE *fout, char *lOpcode, char *lOperand1, char *lOperand2, char *lOperand3, char *lOperand4);

int check_and_write_opcode(FILE *fout, char *lOpcode, char *lOperand1, char *lOperand2, char *lOperand3, char *lOperand4);

int register_match_check(char *lOperand);

int get_PCoffset9(int label_address);

int get_PCoffset11(int label_address);

int toNum(char *pStr);

int searchLabel(char *);

int addLabel(char *);

int isOpcode(char *first_Ptr);

int main(int argc, char* argv[])
{ 
	int indx;
  	infile 		= fopen(argv[1], "r");	
  	outfile 	= fopen(argv[2], "w");
     	 
  	printf("First argument is: %s\n", argv[1]);
  	printf("Second argument is: %s\n", argv[2]);


	if (!infile) {
    	printf("Error: Cannot open file %s\n", argv[1]);
    	exit(4);
  	}
  
  	if (!outfile) {
    	printf("Error: Cannot open file %s\n", argv[2]);
    	exit(4);
  	}

  	/* First pass */
  	first_pass = 1;
  	printf("\tFIRST PASS = %d",first_pass);
  	process_input_output_files(infile, outfile);

  	for(indx=0;indx<labelCounter;indx++)
  	{
    	printf("labelTable[%d].label = %s, labelTable[%d].address = %d\n",indx,labelTable[indx].label,indx,labelTable[indx].address);
  	}

  	rewind(infile);

  	/* second pass */
  	first_pass = 0;
  	printf("\tSECOND PASS\n");
  	process_input_output_files(infile, outfile);
  	fclose(outfile);
}

void process_input_output_files(FILE *fin, FILE *fout)
{
	char lLine[MAX_LINE_LENGTH + 1], *lLabel, *lOpcode, *lArg1,
	        *lArg2, *lArg3, *lArg4;

  	int lRet;

  	printf("process_input_output_files, first_pass = %d",first_pass);
  	
  	do
  	{
      	  lRet = readAndParse( fin, lLine, &lLabel,
          		&lOpcode, &lArg1, &lArg2, &lArg3, &lArg4 );

          	printf("\tLRet\t\t\t\t\t:\t%d\n", lRet);
          printf("\tLine\t\t\t\t\t:\t%s\n", lLine);
          printf("\tLabel\t\t\t\t\t:\t%s\n", lLabel);
          printf("\tOpcode\t\t\t\t\t:\t%s\n", lOpcode);
          printf("\tArg1\t\t\t\t\t:\t%s\n", lArg1);
          printf("\tArg2\t\t\t\t\t:\t%s\n", lArg2);
          printf("\tArg3\t\t\t\t\t:\t%s\n", lArg3);
          printf("\tArg3\t\t\t\t\t:\t%s\n", lArg4);
          printf("\n\n");

          if(!firstLine){
          	if(strcmp(lOpcode, ".orig") == 0){
          		firstLine	= 1;
          		printf("\tCorrect Code, .ORIG found\n");
          	}
          	else if(strcmp(lOpcode, "") == 0){
          		printf("\tEmpty Line\n");
          	}
          	else{
          		printf("\tNo .ORIG available, wrong code\n");
          		exit(4);
          	}
          }


       	  if( lRet != DONE && lRet != EMPTY_LINE )
       	  {
         		if(strcmp(lOpcode,".end") == 0)
         		{
           		lRet = DONE;
         		}
         		else
         		{
           		check_and_write_output(fout, lLabel, lOpcode, lArg1, 
           		     	lArg2, lArg3, lArg4);
           		printf("\tAfter check_and_write_output, LRet\t\t\t\t\t:\t%d\n", lRet);
         		}
       	  }
  	}while( lRet != DONE );
}

/**
 *  Function to read a line of input assembly language file to obtain the 
 *  label, opcode and operands.
 */
   
int readAndParse( FILE *pInfile, char *pLine, char **pLabel, char **pOpcode, char ** pArg1, char ** pArg2, char ** pArg3, char ** pArg4)
{
	char * lRet, * lPtr;
   	int i;

   	if( !fgets( pLine, MAX_LINE_LENGTH, pInfile ) )
			return( DONE );
   
   	for( i = 0; i < strlen( pLine ); i++ )
			pLine[i] = tolower( pLine[i] );
   
   	/* convert entire line to lowercase */
   	*pLabel = *pOpcode = *pArg1 = *pArg2 = *pArg3 = *pArg4 = pLine + strlen(pLine);

   	lPtr = pLine;

   	/* Ignore comments and blank lines.*/
   	while( *lPtr != ';' && *lPtr != '\0' && *lPtr != '\n' ) 
		lPtr++;

   	*lPtr = '\0';
   	if( !(lPtr = strtok( pLine, "\t\n ," ) ) ) 
		return( EMPTY_LINE );

   	if( isOpcode( lPtr ) == -1 && lPtr[0] != '.' ) /* found a label */
   	{ 
		*pLabel = lPtr;
		if( !( lPtr = strtok( NULL, "\t\n ," ) ) ) return( OK );
   	}
   
   	*pOpcode = lPtr;

   	if( !( lPtr = strtok( NULL, "\t\n ," ) ) ) return( OK );
   
   	*pArg1 = lPtr;
   
   	if( !( lPtr = strtok( NULL, "\t\n ," ) ) ) return( OK );

   	*pArg2 = lPtr;
   	if( !( lPtr = strtok( NULL, "\t\n ," ) ) ) return( OK );

   	*pArg3 = lPtr;

   	if( !( lPtr = strtok( NULL, "\t\n ," ) ) ) return( OK );

   	*pArg4 = lPtr;

   	return( OK );
}

void check_and_write_output(FILE *fout, char *lLabel, char *lOpcode, char *lArg1, 
 		       char *lArg2, char *lArg3, char *lArg4)
{
  int indx;
  int Opcode;
  int Label;

  printf("check_and_write_output, first_pass = %d, fout = %d, lLabel = %s, lOpcode = %s, lArg1 = %s, lArg2 = %s, lArg3 = %s, lArg4 = %s\n",first_pass, fout, lLabel, lOpcode, lArg1,lArg2,lArg3,lArg4);

  if(strlen(lLabel) != 0)
  {
    Label = 1;
    printf("\tlLabel is not null, lLabel = %s\n", lLabel);
  }

  if(strlen(lOpcode) != 0)
  {
    Opcode = 1;
    printf("\tLopcode is not null, lOpcode = %s\n", lOpcode);
  }

  if(strlen(lArg1) != 0)
  {
    printf("\tlArg1 is not null, lArg1 = %s\n", lArg1);
  }

  if(strlen(lArg2) != 0)
  {
    printf("\tlArg2 is not null, lArg2 = %s\n", lArg2);
  }

  if(strlen(lArg3) != 0)
  {
    printf("\tlArg3 is not null, lArg3 = %s\n", lArg3);
  }

  if(strlen(lArg4) != 0)
  {
    printf("\tlArg4 is not null, lArg4 = %s\n", lArg4);
  }

  printf("\tcheck_and_write_output, Label = %d, Opcode = %d\n", Label, Opcode);

  if(Label == 1)
  {
    if(first_pass)
    {
      /* Add label to the Symbol table*/
      addLabel(lLabel);
    }
  }

  if(Opcode == 1)
  {
    if(!(isOpcode(lOpcode)))
    {
      printf("\tFirst Pass = %d\n\n\n",first_pass);
      if(first_pass == 0)
      {
        check_and_write_opcode(fout, lOpcode, lArg1, lArg2, lArg3, lArg4);
      }
      else if(first_pass == 1)
      {
        location_counter+=2;
      }
    }
    else if(lOpcode[0] == '.')
    {
      check_and_write_pseudo_op(fout, lOpcode, lArg1, lArg2, lArg3, lArg4);
    }
    /* Neither opcode nor pseudo-op, so error */
    else
    {
      printf("exit 1");
      exit(2);
    }
  }
  else
  {
    printf("exit 2");
    /* A non-empty and non-commented line must have an Opcode/Pseudo-Op. */
    exit(2);
  }

}

int check_and_write_pseudo_op(FILE *fout, char *lOpcode, char *lOperand1, char *lOperand2, char *lOperand3, char *lOperand4){
  int index;
  int string_length;
  int output_int;
  char *lOperand1_ptr;
  int Operand1;
  int Operand2;
  int Operand3;
  int Operand4;

  if(strlen(lOperand1) != 0)
  {
    Operand1 = 1;
    printf("\tlOperan1 is not null, lOperand1 = %s\n", lOperand1);
  }

  if(strlen(lOperand2) != 0)
  {
    Operand2 = 1;
    printf("\tlOperan2 is not null, lOperand2 = %s\n", lOperand2);
  }

  if(strlen(lOperand3) != 0)
  {
    Operand3 = 1;
    printf("\tlOperan3 is not null, lOperand3 = %s\n", lOperand3);
  }

  if(strlen(lOperand4) != 0)
  {
    Operand4 = 1;
    printf("\tlOperan4 is not null, lOperand4 = %s\n", lOperand4);
  }

  if(strcmp(".orig", lOpcode) == 0)
  {
    if(Operand1 == 1)
    {
      /* Initialize Line Counter */
      location_counter = toNum(lOperand1);
      printf("location counter = %d",location_counter);
      /* Check whether address is within the memory address space */
      if(location_counter > pow(2, 16))
      {
	printf("check_and_write_pseudo_op, exit1\n");
        exit(3);
      }
      /* Check whether address is even. */
      else if(location_counter%2 != 0)
      {
        printf("check_and_write_pseudo_op, exit2\n");
        exit(3);
      }
      /* Only one operand should be specified for .orig.
      If yes, WRITE the pseudo-op instruction to the output file.*/
      if(Operand2 != 1)
      {
        if(first_pass == 0)
  	{
          fprintf(fout, "0x%.4X\n", location_counter);
	}
      }
      else
      {
        printf("check_and_write_pseudo_op, exit4\n");
        exit(4);
      } 
    }
    else
    {
      printf("check_and_write_pseudo_op, exit4\n");
      /* No operand supplied to .orig */
      exit(4);
    }
  }
  else if(strcmp(".fill", lOpcode) == 0)
  {
    location_counter +=2;
    /* NOTE:Should there be a check for whether the location that is being asked to be filled
      is out of the acceptable memory space? */
    if(Operand1 == 1)
    {
      /* More no. of arguments than expected */
      if(Operand2 == 1)
      {
        exit(4);
      }
      else
      {
        if(first_pass == 0)
	{
	  output_int = toNum(lOperand1);
          fprintf(fout, "0x%.4X\n", output_int);
	}
      }
    }
    /* missing operand */
    else        
    {
      exit(4);
    }
  }
  else if(strcmp(".blkw", lOpcode) == 0)
  {
    location_counter +=(toNum(lOperand1)*2);
    /* NOTE:Should there be a check for whether the location that is being asked to be filled
       is out of the acceptable memory space? */
    if(Operand1 == 1)
    {
      /* More no. of arguments than expected */
      if(Operand2 == 1)
      {
        exit(4);
      }
      else
      {
        if(first_pass == 0)
	{
          fprintf(fout, "0x%.4X\n", toNum(lOperand1));
	}
      }
    }
    /* missing operand */
    else
    {
      exit(4);
    }
  }
  else if(strcmp(".stringz", lOpcode) == 0)
  {
    lOperand1_ptr = lOperand1;
    /* NOTE:Should there be a check for whether the location that is being asked to be filled
       is out of the acceptable memory space? */
    if(Operand1 == 1)
    {   
      string_length = strlen(lOperand1);

      if(string_length%2 == 0)
      { 
        location_counter = location_counter + string_length + 2;
      }
      else
      {
       location_counter = location_counter + string_length + 1;
      }
      /* More no. of arguments than expected */
      if(Operand2 == 1)
      {
        exit(4);
      }
      else
      {
        /* Check whether operand is a string */
        for(index=0; index<string_length; index++)
  	{
	  if(index == 0 || index == (string_length-1))
          {
 	    if(!(*lOperand1_ptr == '"'))
	    {
	      exit(3);
	    }
          }
          else
	  {
            if(!(isalpha(*lOperand1_ptr)))
            {
	      exit(3);
	    }
          }
          lOperand1_ptr++;
	}
        if(first_pass == 0)
	{
       	  fprintf(fout, "0x%.4X\n", toNum(lOperand1));
	}
      }
    }
    else
    {
      exit(4);
    }
  }
  else if(strcmp(".end",lOpcode) == 0)
  {
    /* .end should have no operand */
    if(Operand1 == 1)
    {
      exit(4);
    }
    else
    {
      exit(0);
    }
  }
  /* Opcode starting with . but not matching an acceptable pseudo-op indicated invalid opcode. */
  else
  {
    exit(2);
  }
}

int check_and_write_opcode(FILE *fout, char *lOpcode, char *lOperand1, char *lOperand2, char *lOperand3, char *lOperand4)
{
  int index;
  int label_address;
  int int_operand;
  int arg_register1;
  int arg_register2;
  int arg_register3;
  int base;
  int label_found = 0;
  int Operand1;
  int Operand2;
  int Operand3;
  int Operand4;

  location_counter+=2;

  if(strlen(lOperand1) != 0)
  {
    Operand1 = 1;
  }

  if(strlen(lOperand2) != 0)
  {
    Operand2 = 1;
  }

  if(strlen(lOperand3) != 0)
  {
    Operand3 = 1;
  }

  if(strlen(lOperand4) != 0)
  {
    Operand4 = 1;
  }

  printf("check_and_write_output");
  if(strcmp("add",lOpcode) == 0)
  {
    if(Operand1 == 1)
    {
      if(Operand2 == 1)
      {
        if(Operand3 == 1)
        {
          /* Check first operand */
          arg_register1 = register_match_check(lOperand1);
          if(arg_register1 != -1)
          {
            /* Check second operand */
            arg_register2 = register_match_check(lOperand2);
	    if(arg_register2 != -1)
	    {
	      /* Check third operand */
	      arg_register3 = register_match_check(lOperand3);
              if(arg_register3 != -1)
              {
		printf("\nADD: register operand");
 	        /* All 3 are registers, so write to output */
		base = (1 << 12) | (arg_register1 << 9) | (arg_register2 << 6) | arg_register3;
		fprintf(fout, "0x%.4X\n", base);
 	      }
              else
              {
	        int_operand = toNum(lOperand3);
		if(!(int_operand < -16 || int_operand > 15))
		{
 		  printf("\nADD: int_operand = %d, %.4x",int_operand,int_operand);
		  base = (1 << 12) | (arg_register1 << 9) | (arg_register2 << 6) | (1 << 5) | (int_operand & 31);
                  fprintf(fout, "0x%.4X\n", base);
		}
		else
		{
		  exit(3);
		}
              }
	    }
	    else/* second operand is not a register */
	    {
	      exit(4);
	    }
          }
	  else/* first operand is not a register */
	  {
	    exit(4);
	  }
	}/* third operand is not null */
	else
	{
	  exit(4);
	}
      }/* second operand is not null */
      else
      {
        exit(4);
      }
    }/* first operand is not null */
    else
    {
      exit(4);
    }
    /** Exit if a fourth Operand is present.*/
    if(Operand4 == 1)
    { 
      exit(4);
    }
  }/* end ADD */
  else if(strcmp("and",lOpcode) == 0)
  {
   if(Operand1 == 1)
    {
      if(Operand2 == 1)
      {
        if(Operand3 == 1)
        {
          arg_register1 = register_match_check(lOperand1);
          if(arg_register1 != -1)
          {
            arg_register2 = register_match_check(lOperand2);
            if(arg_register2 != -1)
            {
              arg_register3 = register_match_check(lOperand3);
              if(arg_register3 != -1)
              {
		              base = (5 << 12) | (arg_register1 << 9) | (arg_register2 << 6) | arg_register3;
                  fprintf(fout, "0x%.4X\n", base);
              }
              else
              {
                int_operand = toNum(lOperand3);
                if(!(int_operand < -16 || int_operand > 15))
                {
		                base = (5 << 12) | (arg_register1 << 9) | (arg_register2 << 6) | (1 << 5) | (int_operand & 31);
                    fprintf(fout, "0x%.4X\n", base);
                }
                else
                {
                  exit(3);
                }
              }
            }
            else
            {
              exit(4);
            }
          }
          else/*first operand is not a register */
          {
            exit(4);
          }
        }/*third operand is not null */
        else
        {
          exit(4);
        }
      }/* second operand is not null */
      else
      {
        exit(4);
      }
    }/* first operand is not null */
    else
    {
      exit(4);
    }
    /** Exit if a fourth Operand is present.*/
    if(Operand4 == 1)
    {
      exit(4);
    }
  }
  else if(strcmp("brz",lOpcode) == 0)
  {
   if(Operand1 == 1)
    {
      /* exit if there are more than valid number of arguments */
      if(Operand2 == 1)
      {
        exit(4);
      }
      else
      {
        /* Check if label is defined */
        for(index=0;index<labelCounter;index++)
        {
          if(strcmp(labelTable[index].label,lOperand1) == 0)
          {
            label_found = 1;
            label_address = labelTable[index].address;
            break;
          }
        }
        if(!label_found)
        {
          exit(1);
        }
        else
        {
	  base = (1 << 10) | ((get_PCoffset9(label_address)) & 511);
          fprintf(fout, "0x%.4X\n", base);
        }
      }
    }
    else
    {
      exit(4);
    }
  }
  else if(strcmp("brp", lOpcode) == 0)
  {
   if(Operand1 == 1)
    {
      if(Operand2 == 1)
      {
        exit(4);
      }
      else
      {
        for(index=0;index<labelCounter;index++)
        {
          if(strcmp(labelTable[index].label,lOperand1) == 0)
          {
            label_found = 1;
            label_address = labelTable[index].address;
            break;
          }
        }
        if(!label_found)
        {
          exit(1);
        }
        else
        {
	  base = (1 << 9) | ((get_PCoffset9(label_address)) & 511);
          fprintf(fout, "0x%.4X\n", base);
        }
      }
    }
    else
    {
      exit(4);
    }
  }
  else if(strcmp("brn", lOpcode) == 0)
  {
   if(Operand1 == 1)
    {
      if(Operand2 == 1)
      {
        exit(4);
      }
      else
      {
        for(index=0;index<labelCounter;index++)
        {
          if(strcmp(labelTable[index].label,lOperand1) == 0)
          {
            label_found = 1;
            label_address = labelTable[index].address;
            break;
          }
        }
        if(!label_found)
        {
          exit(1);
        }
        else
        {
	  base = (1 << 11) | ((get_PCoffset9(label_address)) & 511);
          fprintf(fout, "0x%.4X\n", base);
        }
      }
    }
    else
    {
      exit(4);
    }
  }
  else if(strcmp("brnz", lOpcode) == 0)
  {
   if(Operand1 == 1)
    {
      if(Operand2 == 1)
      {
        exit(4);
      }
      else
      {
        for(index=0;index<labelCounter;index++)
        {
          if(strcmp(labelTable[index].label,lOperand1) == 0)
          {
            label_found = 1;
            label_address = labelTable[index].address;
            break;
          }
        }
        if(!label_found)
        {
          exit(1);
        }
        else
        {
	  base = (3 << 10) | ((get_PCoffset9(label_address)) & 511);
          fprintf(fout, "0x%.4X\n", base);
        }
      }
    }
    else
    {
      exit(4);
    }
  }
  else if(strcmp("brzp", lOpcode) == 0)
  {
    if(Operand1 == 1)
    {
      if(Operand2 == 1)
      {
        exit(4);
      }
      else
      {
        for(index=0;index<labelCounter;index++)
        {
          if(strcmp(labelTable[index].label,lOperand1) == 0)
          {
            label_found = 1;
            label_address = labelTable[index].address;
            break;
          }
        }
        if(!label_found)
        {
          exit(1);
        }
        else
        {
	  base = (3 << 9) | ((get_PCoffset9(label_address)) & 511);
          fprintf(fout, "0x%.4X\n", base);
        }
      }
    }
    else
    {
      exit(4);
    }
  }
  else if(strcmp("brnp", lOpcode) == 0)
  {
    if(Operand1 == 1)
    {
      if(Operand2 == 1)
      {
        exit(4);
      }
      else
      {
        for(index=0;index<labelCounter;index++)
        {
          if(strcmp(labelTable[index].label,lOperand1) == 0)
          {
            label_found = 1;
            label_address = labelTable[index].address;
            break;
          }
        }
        if(!label_found)
        {
          exit(1);
        }
        else
        {
	  base = (5 << 9) | ((get_PCoffset9(label_address)) & 511);
          fprintf(fout, "0x%.4X\n", base);
        }
      }
    }
    else
    {
      exit(4);
    }
  }
  else if(strcmp("brnzp", lOpcode) == 0)
  {
    if(Operand1 == 1)
    {
      if(Operand2 == 1)
      {
        exit(4);
      }
      else
      { 
        for(index=0;index<labelCounter;index++)
        {
          if(strcmp(labelTable[index].label,lOperand1) == 0)
          {
            label_found = 1;
            label_address = labelTable[index].address;
            break;
          }
        }
        if(!label_found)
        {
          exit(1);
        }
        else
        {
	  base = (7 << 9) | ((get_PCoffset9(label_address)) & 511);
          fprintf(fout, "0x%.4X\n", base);
        }
      }
    }
    else
    {
      exit(4);
    }
  }
  else if(strcmp("br", lOpcode) == 0)
  {
    if(Operand1 == 1)
    {
      if(Operand2 == 1)
      {
        exit(4);
      }
      else
      {
        for(index=0;index<labelCounter;index++)
        {
          if(strcmp(labelTable[index].label,lOperand1) == 0)
          {
            label_found = 1;
            label_address = labelTable[index].address;
            break;
          }
        }
        if(!label_found)
        {
          exit(1);
        }
        else
        {
	  base = ((get_PCoffset9(label_address)) & 511);
          fprintf(fout, "0x%.4X\n", base);
        }
      }
    }
    else
    {
      exit(4);
    }
  }
  else if(strcmp("halt", lOpcode) == 0)
  {
    /* HALT does not take any operand */
    if(Operand1 == 1)
    {
      exit(4);
    }
    else
    {
      base = 61477; /* Int value for b'1111 0000 0010 0101*/
      fprintf(fout, "0x%.4X\n", base);      
    }
  }
else if(strcmp("jmp", lOpcode) == 0)
{
        if(Operand1 == 1)
        {
                arg_register1 = register_match_check(lOperand1);
                /* Register Operand */
                if(arg_register1 != -1)
                {
                        if(Operand2 == 1)
                        {
                                exit(4);
                        }
                        else
                        {
                                base = (3<<14) | (arg_register1 << 6);
                                fprintf(fout, "0x%.4X\n", base);
                        }
                }
                else
                {
                        exit(4);
                }
        }
        else
        {
                /* No arguments to JMP */
                exit(4);
        }
}

else if(strcmp("jsr", lOpcode) == 0)
{
        if(Operand1 == 1)
        {
                if(Operand2 == 1)
                {
                        exit(4);
                }
                else
                {
                        for(index=0;index<labelCounter;index++)
                        {
                                if(strcmp(labelTable[index].label,lOperand1) == 0)
                                {
                                        label_found = 1;
                                        label_address = labelTable[index].address;
                                        break;
                                }
                        }
                        if(!label_found)
                        {
                                exit(1);
                        }
                        else
                        {
                                base = (4 << 12) | (1 << 11) | ((get_PCoffset11(label_address)) & 2047);
                                fprintf(fout, "0x%.4X\n", base);
                        }
                }
        }
        else
        {
                exit(4);
        }
}

else if(strcmp("jsrr", lOpcode) == 0)
{
        if(Operand1 == 1)
        {
                arg_register1 = register_match_check(lOperand1);
                if(arg_register1 != -1)
                {
                /* Check that there is no second operand */
                        if(Operand2 == 1)
                        {
                                exit(4);
                        }
                        else
                        {
                                base = (4 << 12) | (arg_register1 << 6);
                                fprintf(fout, "0x%.4X\n", base);
                        }
                }
                else
                {
                        exit(4);
                }
        }
        else
        {
        /* No arguments to JSRR */
                exit(4);
        }
}

else if(strcmp("ldb", lOpcode) == 0)
{
        if(Operand1 == 1)
        {
                if(Operand2 == 1)
                {
                        if(Operand3 == 1)
                        {
			  if(Operand4 == 1)
			  {
			    exit(4);
			  }
			  else
			  {
                                arg_register1 = register_match_check(lOperand1);
                                if(arg_register1 != -1)
                                {
                                        arg_register2 = register_match_check(lOperand2);
                                        if(arg_register2 != -1)
                                        {
                                                int_operand = toNum(lOperand3);
                                                /* Check to see whether offset value is within range */
                                                if(int_operand < -32 || int_operand > 31)
                                                {
                                                        exit(3);
                                                }
                                                else
                                                {
                                                        base = (2 << 12) | (arg_register1 << 9) | (arg_register2 << 6) | (int_operand & 63);
                                                        fprintf(fout, "0x%.4X\n", base);
                                                }
                                        }
                                        else/* second operand is not a register */
                                        {
                                                exit(4);
                                        }
                                }
                                else/* first operand is not a register */
                                {
                                        exit(4);
                                }
			  }
                        }
                        /* third operand is not null */
                        else
                        {
                                exit(4);
                        }
                }/* second operand is not null */
                else
                {
                        exit(4);
                }
        }
        else
        {
                exit(4);
        }
}


else if(strcmp("ldw", lOpcode) == 0){
  	
  	if(Operand1 == 1){
     
      	        if(Operand2 == 1){

                        if(Operand3 == 1)
                        {
			    if(Operand4 == 1)
			    {
			      exit(4);
			    }
			    else
			    {
                                arg_register1 = register_match_check(lOperand1);
                                printf("\tArg Register 1 received %d\n", arg_register1);

                                if(arg_register1 != -1){
                                        arg_register2 = register_match_check(lOperand2);
                                        printf("\tArg Register 2 received %d\n", arg_register2);

                                        if(arg_register2 != -1){
                                                int_operand = toNum(lOperand3);
                                                /* Check to see whether offset value is within range */
                                                if(int_operand < -32 || int_operand>31){ 
                                                        exit(3);             
                                                }
                                                else{
                                                        base = (6 << 12) | (arg_register1 << 9) | (arg_register2 << 6) | (int_operand & 63);
                                                        fprintf(fout, "0x%.4X\n", base);
                                                }
                                        }
                                        else/* second operand is not a register */{
                                                exit(4);
                                        }
				
                                }
                                else{/* first operand is not a register */
                                        exit(4);
                                }
			    }
                        }/* third operand is not null */
                        else{
                                exit(4);
                        }/* second operand is not null */
                }
                else{
                       exit(4);
                }
        }/* first operand is not null */
        else
        {
                exit(4);
        }   
}

else if(strcmp("lea", lOpcode) == 0)
{
        if(Operand1 == 1)
        {
                if(Operand2 == 1)
                {
                        if(Operand3 == 1)
                        {
                                exit(4);
			}
			else
			{
                                arg_register1 = register_match_check(lOperand1);
                                if(arg_register1 != -1)
                                {
                                        for(index=0;index<labelCounter;index++)
                                        {
                                                if(strcmp(labelTable[index].label,lOperand2) == 0)
                                                {
                                                        label_found = 1;
                                                        label_address = labelTable[index].address;
                                                        break;
                                                }
                                        }
                                        if(!label_found)
                                        {
                                                exit(1);
                                        }
                                        else
                                        {       
                                                base = (14 << 12) | (arg_register1 << 9) | ((get_PCoffset9(label_address)) & 511);
                                                fprintf(fout, "0x%.4X\n", base);
                                        }
                                }
                                else/* first operand is not a register */
                                {
                                        exit(4);
                                }
                        }/* third operand is not null */
                }/* second operand is not null */
                else
                {
                        exit(4);
                }
 	}/* first operand is not null */
        else
        {
                exit(4);
        }
}

else if(strcmp("nop", lOpcode) == 0)
{
        if(Operand1 == 1)
        {
        /* No operand expected */
                exit(4);
        }
        else
        {
                fprintf(fout,"0x0000\n");
        }
}

else if(strcmp("not", lOpcode) == 0)
{
        if(Operand1 == 1)
        {
                if(Operand2 == 1)
                {
                        if(Operand3 == 1)
                        {/* More than the expected number of arguments. */
                                exit(4);
                        }
                        else
                        {
                                arg_register1 = register_match_check(lOperand1);
                                if(arg_register1 != -1)
                                {
                                        arg_register2 = register_match_check(lOperand2);
                                        if(arg_register2 != -1)
                                        {
                                                base = (9 << 12) | (arg_register1 << 9) | (arg_register2 << 6) | 63;
                                                fprintf(fout, "0x%.4X\n", base);
                                        }
                                        else
                                        {/* Second operand is not a register */
                                                exit(4);
                                        }
                                }
                                else
                                {
                                        exit(4);
                                }
                        }
                }
                else
                {
                /* Inadequate number of operands */
                        exit(4);
                }
        }
        else
        {
                exit(4);
        }
}

else if(strcmp("ret", lOpcode) == 0)
{
/* Exit if RET has an operand */
        if(Operand1 == 1)
        {
                exit(4);
        }
        else
        {
                base = 49600; /* 'b110000011100000 */
                fprintf(fout, "0x%.4X\n", base);
        }
}

else if(strcmp("lshf", lOpcode) == 0)
{
        if(Operand1 == 1)
        {
                if(Operand2 == 1)
                {
                        if(Operand3 == 1)
                        {
			  if(Operand4 == 1)
			  {
			    exit(4);
			  }
			  else
			  {
                                arg_register1 = register_match_check(lOperand1);
                                if(arg_register1 != -1)
                                {
                                        arg_register2 = register_match_check(lOperand2);
                                        if(arg_register2 != -1)
                                        {
                                                int_operand = toNum(lOperand3);
                                                /*Check range of amount4 */
                                                if(int_operand < 0 || int_operand > 15)
                                                {
                                                        exit(3);
                                                }
                                                else
                                                {
                                                        base = (13 << 12) | (arg_register1 << 9) | (arg_register2 << 6) | (int_operand);
                                                        fprintf(fout, "0x%.4X\n", base);
                                                }
                                        }
                                        else/* second operand is not a register */
                                        {
                                                exit(4);
                                        }
                                }
                                else/* first operand is not a register */
                                {
                                        exit(4);
                                }
			  }
                        }/* third operand is not null */
                        else
                        {
                                exit(4);
                        }
                }/* second operand is not null */
                else
                {
                        exit(4);
                }
        }/* first operand is not null */
        else
        {
                exit(4);
        }
}

else if(strcmp("rshfa", lOpcode) == 0)
{
        if(Operand1 == 1)
        {
                if(Operand2 == 1)
                {
                        if(Operand3 == 1)
                        {
                          if(Operand4 == 1)
                          {
                            exit(4);
                          }
                          else
                          {
                                arg_register1 = register_match_check(lOperand1);
                                if(arg_register1 != -1)
                                {
                                        arg_register2 = register_match_check(lOperand2);
                                        if(arg_register2 != -1)
                                        {
                                                int_operand = toNum(lOperand3);
                                                /*Check range of amount4 */
                                                if(int_operand < 0 || int_operand > 15)
                                                { 
                                                        exit(3);
                                                } 
                                                else
                                                {
                                                        base = (13 << 12) | (arg_register1 << 9) | (arg_register2 << 6) | (3 << 4) | int_operand;
                                                        fprintf(fout, "0x%.4X\n", base);
                                                }
                                        }
                                        else
                                        {
                                                exit(4);
                                        }
                                }
                                else
                                {
                                        exit(4);
                                }
			  }
                        }
                        else
                        {
                                exit(4);
                        }
                }
                else
                {
                        exit(4);
                }
        }
        else
        {
                exit(4);
        }
}

else if(strcmp("rshfl", lOpcode) == 0)
{
        if(Operand1 == 1)
        {
                if(Operand2 == 1)
                {
                        if(Operand3 == 1)
                        {
                          if(Operand4 == 1)
                          {
                            exit(4);
                          }
                          else
                          {
                                arg_register1 = register_match_check(lOperand1);
                                if(arg_register1 != -1)
                                {
                                        arg_register2 = register_match_check(lOperand2);
                                        if(arg_register2 != -1)
                                        {
                                                int_operand = toNum(lOperand3);
                                                /*Check range of amount4 */
                                                if(int_operand < 0 || int_operand > 15)
                                                {
                                                        exit(3);
                                                }
                                                else
                                                {
                                                        base = (13 << 12) | (arg_register1 << 9) | (arg_register2 << 6) | (1 << 4) | int_operand;
                                                        fprintf(fout, "0x%.4X\n", base);
                                                }
                                        }
                                        else
                                        {
                                                exit(4);
                                        }
                                }
                                else
                                {
                                        exit(4);
                                }
			  }
                        }
                        else
                        {
                                exit(4);
                        }
                }
                else
                {
                        exit(4);
                }
        }
        else
        {
                exit(4);
        }
}


else if(strcmp("rti", lOpcode) == 0)
{
        if(Operand1 == 1)
        {
                exit(4);
        }
        else
        {
                fprintf(fout, "0x8000\n");
        }
}

else if(strcmp("stb", lOpcode) == 0)
{
        if(Operand1 == 1)
        {
                if(Operand2 == 1)
                {
                        if(Operand3 == 1)
                        {
                          if(Operand4 == 1)
                          {
                            exit(4);
                          }
                          else
                          {
                                arg_register1 = register_match_check(lOperand1);
                                if(arg_register1 != -1)
                                {
                                        arg_register2 = register_match_check(lOperand2);
                                        if(arg_register2 != -1)
                                        {
                                                int_operand = toNum(lOperand3);
                                                /* Check to see whether offset value is within range */
                                                if((int_operand < -32) || (int_operand > 31))
                                                {
                                                        exit(3);
                                                }
                                                else
                                                {
                                                        base = (3 << 12) | (arg_register1 << 9) | (arg_register2 << 6) | (int_operand & 63);
                                                        fprintf(fout, "0x%.4X\n", base);
                                                }
                                        }
                                        else
                                        {
                                                exit(4);
                                        }
                                }
                                else
                                {
                                        exit(4);
                                }
			  }
                        }
                        else
                        {
                                exit(4);
                        }
                }
                else
                { 
                        exit(4);
                }
        }
        else
        {
                exit(4);
        }
}


else if(strcmp("stw", lOpcode) == 0)
{
        if(Operand1 == 1)
        {
                if(Operand2 == 1)
                {
                        if(Operand3 == 1)
                        {
                          if(Operand4 == 1)
                          {
                            exit(4);
                          }
                          else
                          {
                                arg_register1 = register_match_check(lOperand1);
                                if(arg_register1 != -1)
                                {
                                        arg_register2 = register_match_check(lOperand2);
                                        if(arg_register2 != -1)
                                        {
                                                int_operand = toNum(lOperand3);
                                                /* Check to see whether offset value is within range */
                                                if((int_operand < -32) || (int_operand > 31))
                                                {
                                                        exit(3);
                                                }
                                                else
                                                {
                                                        base = (7 << 12) | (arg_register1 << 9) | (arg_register2 << 6) | (int_operand & 63);
                                                        fprintf(fout, "0x%.4X\n", base);
                                                }
                                        }
                                        else
                                        {
                                                exit(4);
                                        }
                                }
                                else/* first operand is not a register */
                                {
                                        exit(4);
                                }
			  }
                        }/* third operand is not null */
                        else
                        {
                                exit(4);
                        }
                }/* second operand is not null */
                else
                {
                        exit(4);
                }
        }/* first operand is not null */
        else
        {
                exit(4);
        }
}

else if(strcmp("trap", lOpcode) == 0)
{
        if(Operand1 == 1)
        {
                if(Operand2 == 1)
                {
                /* More than expected arguments */
                        printf("\nTRAP, exit 1");
                        exit(4);
                }
                else
                {
                /* Trapvector should be non negative, hexadecimal*/
                        int_operand = toNum(lOperand1);
                        if(int_operand<0 || *lOperand1 != 'x')
                        {
                                printf("\nTRAP, exit 2");
                                exit(3);
                        }
                        else if((int_operand*2)> 510)
                        {
                                printf("\nTRAP, exit 3");
                                exit(3);
                        }
                        else
                        {
                                lOperand1++;
                                fprintf(fout, "0x%X%s\n",240,lOperand1);
                        }
                }
        }
        else/* No trapvector8 */
        {
                printf("\nTRAP, exit 4\n");
                exit(4);
        }
}

else if(strcmp("xor", lOpcode) == 0)
  {
    if(Operand1 == 1)
    {
      if(Operand2 == 1)
      {
        if(Operand3 == 1)
        {
          if(Operand4 == 1)
	  {
 	    exit(4);
	  }
	  else
	  {
            arg_register1 = register_match_check(lOperand1);
            if(arg_register1 != -1)
            {
              arg_register2 = register_match_check(lOperand2);
              if(arg_register2 != -1)
              {
                arg_register3 = register_match_check(lOperand3);
                if(arg_register3 != -1)
                {
	          base = (9 << 12) | (arg_register1 << 9) | (arg_register2 << 6) | arg_register3; 
                  fprintf(fout, "0x%.4X\n", base);
                }
                else
                {
                  int_operand = toNum(lOperand3);
                  if(!(int_operand < -16 || int_operand > 15))
                  {
	            base = (9 << 12) | (arg_register1 << 9) | (arg_register2 << 6) | (1 << 5) | (int_operand & 31);
                    fprintf(fout, "0x%.4X\n", base);
                  }
                  else
                  {
                    exit(3);
                  }
                }
              }
              else/* second operand is not a register */
              {
                exit(4);
              }
            }
            else/* first operand is not a register */
            {
              exit(4);
            }
	  }
        }/* third operand is not null */
        else
        {
          exit(4);
        }
      }/* second operand is not null */
      else
      {
        exit(4);
      }
    }/* first operand is not null */
    else
    {
      exit(4);
    }
  }
  else
  {
    /* Since the string in lOpcode did not match a valid Opcode. */
    exit(2);
  }
}

int register_match_check(char *lOperand){
	if(!strcmp("r0",lOperand))
  	{
  		printf("\tRegister Number 0 Identified\n\n");
    	return(0);
  	}
  	else if(!strcmp("r1",lOperand))
  	{
  		printf("\tRegister Number 1 Identified\n\n");
    	return(1);
  	}
  	else if(!strcmp("r2",lOperand))
  	{
  		printf("\tRegister Number 2 Identified\n\n");
    	return(2);
  	}
  	else if(!strcmp("r3",lOperand))
  	{
  		printf("\tRegister Number 3 Identified\n\n");
    	return(3);
  	}
  	else if(!strcmp("r4",lOperand))
  	{
  		printf("\tRegister Number 4 Identified\n\n");
    	return(4);
  	}
  	else if(!strcmp("r5",lOperand))
  	{
  		printf("\tRegister Number 5 Identified\n\n");
   		return(5);
  	}
  	else if(!strcmp("r6",lOperand))
  	{
  		printf("\tRegister Number 6 Identified\n\n");
    	return(6);
  	}
  	else if(!strcmp("r7",lOperand))
  	{
  		printf("\tRegister Number 7 Identified\n\n");
    	return(7);
  	}
  	else
  	{
  		printf("\tNo Valid Register Number Identified\n\n");
    	return(-1);
  	}
}

int get_PCoffset9(int label_address)
{
  int pcoffset_9;
  pcoffset_9 = (label_address - location_counter)/2;
  
  /* Check to see whethet the value of the offset can be expressed using 9 bits i.e.
  the value of the offset should be between -256 and +255. */

  if((pcoffset_9>255) || (pcoffset_9<-256))
  {
    exit(4);
  }

  return(pcoffset_9);
}

int get_PCoffset11(int label_address)
{
  int pcoffset_11;
  pcoffset_11 = (label_address - location_counter)/2;

  /*Check to see whethet the value of the offset can be expressed using 11 bits i.e.
    the value of the offset should be between -256 and +255. */

  if((pcoffset_11>1023) || (pcoffset_11< -1024))
  {
    exit(4);
  }

  return(pcoffset_11);
}

int toNum(char *pStr)
{
   char * t_ptr;
   char * orig_pStr;
   int t_length,k,carry = 1;
   int lNum, lNeg = 0;
   long int lNumLong;

   orig_pStr = pStr;
   if( *pStr == '#' )				/* decimal */
   { 
     pStr++;
     if( *pStr == '-' )				/* dec is negative */
     {
       lNeg = 1;
       pStr++;
     }
     t_ptr = pStr;
     t_length = strlen(t_ptr);
     for(k=0;k < t_length;k++)
     {
       if (!isdigit(*t_ptr))
       {
	 printf("Error: invalid decimal operand, %s\n",orig_pStr);
	 exit(4);
       }
       t_ptr++;
     }
     lNum = atoi(pStr);
     if (lNeg)
       lNum = -lNum;
 
     return lNum;
   }
   else if( *pStr == 'x' )	/* hex     */
   {
     pStr++;
     if( *pStr == '-' )				/* hex is negative */
     {
       lNeg = 1;
       pStr++;
     }
     t_ptr = pStr;
     t_length = strlen(t_ptr);
     for(k=0;k < t_length;k++)
     {
       if (!isxdigit(*t_ptr))
       {
	 printf("Error: invalid hex operand, %s\n",orig_pStr);
	 exit(4);
       }
       t_ptr++;
     }
     lNumLong = strtol(pStr, NULL, 16);    /* convert hex string into integer */
     lNum = (lNumLong > INT_MAX)? INT_MAX : lNumLong;
     if( lNeg )
       lNum = -lNum;
     return lNum;
   }
   else
   {
	printf( "Error: invalid operand, %s\n", orig_pStr);
	exit(4);  /* This has been changed from error code 3 to error code 4, see clarification 12 */
   }
}

int findLabel(char *fLabel)
{

  printf("\tfindLabel Function Receiving Value\t:\t%s\n", fLabel);
  int i;

  for(i = 0; i < labelCounter; i++)
  {
    if(strcmp(fLabel, labelTable[i].label) == 0)
    {
      printf("\tDuplicate Labels Found");
      exit(4);
    }
  }

  return 0;
}

int addLabel(char *fLabel)
{
  printf("\taddLabel Function Receiving Value\t:\t%s\n", fLabel);

  printf("\tValue of findLabel(fLabel)\t\t:\t%d\n", findLabel(fLabel));

  int length_label;

  int index;

  LabelEntry tempEntry;

  char firstChar;

  firstChar = fLabel[0];

  if(strlen(fLabel) > 20)
  {
    printf("\texit 3\n");
    exit(4);
  }
  else if((!isalpha(fLabel[0])) || (fLabel[0] == 'x'))
  {
    printf("\texit 4\n");
    exit(4);
  }
  else
  {
    length_label = strlen(fLabel);
    for(index = 0; index < length_label; index++)
    {
      if(!(isalpha(fLabel[index]) || isdigit(fLabel[index])))
      {
        printf("\texit 5\n");
        exit(4);
      }
    }
  }

  for(index=0;index<RESTRICTED_LABEL_VALUES_COUNT;index++)
  {
    if(!strcmp(fLabel,restricted_label_values[index]))
    {
      exit(4);
    }
  }

  if(findLabel(fLabel) == 0)
  {
    strcpy(tempEntry.label, fLabel);
    tempEntry.address               = location_counter;

    labelTable[labelCounter]        = tempEntry;

    labelCounter                    = labelCounter + 1;


    printf("\tLabel Counter\t\t\t\t:\t%d\n", labelCounter);
    printf("\tTemp Label\t\t\t\t:\t%s\n", tempEntry.label);
    printf("\tTemp Addr\t\t\t\t:\t%d\n", tempEntry.address);
  }

  return 0;
}

int isOpcode(char * fCode)
{
  int i;
  for(i =0; i < MAX_OPCODE_COUNT; i++)
  {
    if(!strcmp(fCode, valid_opcodes[i]))
    {
      return 0;
    }
  }
  return -1;
}


