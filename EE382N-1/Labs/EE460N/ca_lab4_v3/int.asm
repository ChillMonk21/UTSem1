		.ORIG x1200
		ADD R6, R6, x-2
		STW R1, R6, x0
		ADD R6, R6, x-2
		STW R2, R6, x0
		LEA R1, ADDR
		LDW R1, R1, x0
		LDW R2, R1, x0
		ADD R2, R2, x1
		STW R2, R1, x0
		LDW R2, R6, x0
		ADD R6, R6, x2
		LDW R1, R6, x0
		ADD R6, R6, x2
		RTI
ADDR		.FILL x4000
		.END
