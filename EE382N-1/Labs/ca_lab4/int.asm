		.ORIG x1200
		ADD R6, R6, x-2
		STW R0, R6, x0
		ADD R6, R6, x-2
		STW R1, R6, x0
		LEA R0, ADDR
		LDW R0, R0, x0
		LDW R1, R0, x0
		ADD R1, R1, x1
		STW R1, R0, x0
		LDW R1, R6, x0
		ADD R6, R6, x2
		LDW R0, R6, x0
		ADD R6, R6, x2
		RTI
ADDR		.FILL x4000
		.END
