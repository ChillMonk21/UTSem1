				.ORIG x4000
				AND R7, R7, #0
				LEA R0, L1
				LDB	R7, R0, #1
				LDW R6, R0, #1
				ADD R5, R6, R7
				XOR R4, R7, R6
				NOT R3, R4
				LSHF R1, R3, #1
				RSHFA R1, R3, #1
				RSHFL R1, R3, #1
				AND R2, R4, R5
				STB R2, R0, #3
				STW R3, R0, #3
				HALT
L1				.FILL x-51
				.FILL x230F
				.END
