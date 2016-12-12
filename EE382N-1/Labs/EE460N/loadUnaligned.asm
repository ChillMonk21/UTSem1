				.ORIG x4000
				AND R2, R1, x1
				BRnz UNALIGNED
		
				LDW	R0, R1, x0	
				RET

UNALIGNED		ADD R2, R1,	x0 
				LDB	R3, R2, x0
				ADD R2, R2, x1
				LDB R4, R2, x0
				LSHF R4, R4, x8
				ADD R0, R3, R4
				RET