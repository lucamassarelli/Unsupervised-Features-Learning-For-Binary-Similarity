# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#


# Questa classe estra da un blocco di codice assembler le features utilizzate nell'articolo ccs17

class BlockFeaturesExtractor:
    x86_ARIT = 0
    x86_MOV = 0
    string = []
    dyn_string = []
    constants = []
    num_transfer = 0
    num_instructions = 0
    num_calls = 0
    num_arith = 0

    def __init__(self, architecture, instructions, r2_disasm, string_addr):
        self.architecture = architecture
        self.instructions = instructions
        self.r2_disasm = r2_disasm
        self.string_addr = string_addr

        self.string = []
        self.constant = []
        self.num_transfer = 0
        self.num_instructions = 0
        self.num_calls = 0
        self.num_arith = 0

    def getFeatures(self):
        if len(self.instructions) != 0:
            self.num_instructions = len(self.instructions)
            self.constant, self.string = self.extractConstansStrings()
            self.num_transfer = self.countTransfer()
            self.num_calls = self.countCalls()
            self.num_arith = self.countArith()

        return ({'string': self.string, 'constant': self.constant,
                'transfer': self.num_transfer, 'instruction': self.num_instructions,
                'call': self.num_calls, 'arith': self.num_arith})

    def countCalls(self):
        x86_mnemonics = ['call', 'int']
        arm_mnemonics = ['bl', 'blx']
        mips_mnemonics = ['jal', 'jalr', 'syscall']

        mips_mnemonics = [s.lower() for s in mips_mnemonics]
        arm_mnemonics = [s.lower() for s in arm_mnemonics]
        x86_mnemonics = [s.lower() for s in x86_mnemonics]

        count = 0
        for i in list(self.instructions):
            if self.architecture == 'x86':
                if str(i['mnemonic']) in x86_mnemonics:
                    count = count + 1
            elif self.architecture == 'mips':
                if str(i['mnemonic']) in mips_mnemonics:
                    count = count + 1
            elif self.architecture == 'arm':
                if str(i['mnemonic']) in arm_mnemonics:
                    count = count + 1
        return count

    # Questa funzione conta le istruzione aritmetiche all'interno del blocco
    def countArith(self):
        x86_mnemonics = ['add', 'sub', 'div', 'imul', 'idiv', 'mul', 'shl', 'dec', 'adc', 'adcx', 'addpd', 'addps',
                         'addsd', 'addss', 'addsubpd', 'ADDSUBPS', 'adox', 'divpd', 'divps'
            , 'divsd', 'divss', 'dppd', 'dpps', 'f2xm1', 'fabs', 'fadd', 'faddp', 'fcos', 'fdiv', 'fdivp', 'fiadd',
                         'fidiv', 'fimul', 'fisub', 'fisubr', 'fmul', 'fmulp', 'FPATAN', 'FPREM', 'FPREM1', 'FPTAN',
                         'FRNDINT', 'FSCALE'
            , 'FSIN', 'FSINCOS', 'FSQRT', 'FSUB', 'FSUBP', 'FSUBR', 'FSUBRP', 'FYL2X', 'FYL2XP1', 'HADDPD', 'HADDPS',
                         'HSUBPD', 'HSUBPS', 'KADDB', 'KADDD', 'KADDD', 'KADDW', 'KSHIFTLB', 'KSHIFTLD', 'KSHIFTLQ',
                         'KSHIFTLW', 'KSHIFTRB', 'KSHIFTRD', 'KSHIFTRQ', 'KSHIFTRW'
            , 'MAXPD', 'MAXPS', 'MAXSD', 'MAXSS', 'MINPD', 'MINPS', 'MINSD', 'MINSS', 'MULPD'
            , 'MULPS', 'MULSS', 'MULSD', 'MULX', 'PADDB', 'PADDD', 'PADDQ', 'PADDSB', 'PADDSW', 'PADDUSB', 'PADDUSW'
            , 'PADDW', 'PAVGB', 'PAVGW', 'PHADDD', 'PHADDSW', 'PHADDW', 'PHMINPOSUW', 'PHSUBD', 'PHSUBSW', 'PHSUBW'
            , 'PMADDUBSW', 'PMADDWD', 'PMAXSB', 'PMAXSD', 'PMAXSQ', 'PMAXSW', 'PMAXUB', 'PMAXUD', 'PMAXUQ', 'PMAXUW',
                         'PMINSB'
            , 'PMINSD', 'PMINSQ', 'PMINSW', 'PMINUB', 'PMINUD', 'PMINUQ', 'PMINUW', 'PMULDQ', 'PMULHRSW', 'PMULHUW',
                         'PMULHW', 'PMULLD', 'PMULLQ'
            , 'PMULLW', 'PMULUDQ', 'PSADBW', 'PSLLD', 'PSLLW', 'PSRAD', 'PSLLQ', 'PSRAQ', 'PSRLQ', 'PSRLW'
            , 'PSUBB', 'PSUBD', 'PSUBQ', 'PSUBSB', 'PSUBSW', 'PSUBUSB', 'PSUBUSW', 'RCL', 'RCR'
            , 'ROL', 'ROR', 'ROUNDPD', 'ROUNDPS', 'ROUNDSD', 'ROUNDSS', 'RSQRTPS'
            , 'RSQRTSS', 'SAL', 'SAR', 'SARX', 'SBB', 'inc', 'SHLD', 'SHLX', 'SHR', 'SHRD', 'SHRX', 'SQRTPD', 'SQRTPS',
                         'SQRTSD', 'SQRTSS'
            , 'SUBPD', 'SUBPS', 'SUBSD', 'SUBSS', 'VFMADD132PD', 'VPSLLVD', 'VPSLLVQ', 'VPSLLVW', 'VPSRAVD', 'VPSRAVQ'
            , 'VPSRAVW', 'VPSRLVD', 'VPSRLVQ', 'VPSRLVW', 'VRNDSCALEPD', 'VRNDSCALEPS', 'XADD']

        arm_mnemonics = ['add', 'adc', 'qadd', 'dadd', 'sub', 'SBC', 'RSB', 'RSC', 'subs', 'qsub',
                         'add16', 'SUB16', 'add8', 'sub8', 'ASX', 'sax', 'usad8', 'SSAT', 'MUL'
            , 'smul', 'MLA', 'MLs', 'UMULL', 'UMLAL', 'UMaAL', 'SMULL', 'smlal'
            , 'SMULxy', 'SMULWy', 'SMLAxy', 'SMLAWy', 'SMLALxy', 'SMUAD'
            , 'SMLAD', 'SMLALD', 'SMUSD', 'SMLSD', 'SMLSLD', 'SMMUL'
            , 'SMMLA', 'MIA', 'MIAPH', 'MIAxy', 'SDIV', 'udiv'
            , 'ASR', 'LSL', 'LSR', 'ROR', 'RRX']

        mips_mnemonics = ['add', 'addu', 'addi', 'addiu', 'mult', 'multu', 'div', 'divu'
            , 'AUI', 'DAUI', 'DAHI', 'DATI', 'CLO', 'CLZ', 'DADD', 'DADDI'
            , 'DADDIU', 'DADDU', 'DCLO', 'DCLZ', 'DDIV', 'DDIVU', 'MOD'
            , 'MODU', 'DMOD', 'DMODU', 'DMULTU', 'DROTR', 'DROTR32', 'DSLLV'
            , 'DSRA', 'DSRA32', 'DSRAV', 'DSRL', 'DSRL32'
            , 'DSRLV', 'DSUB', 'DSUBU', 'DSRL', 'FLOOR', 'MAX', 'MIN', 'MINA', 'MAXA'
            , 'MSUB', 'MSUBU', 'MUL', 'MUH', 'MULU', 'MUHU', 'DMUL', 'DMUH'
            , 'DMULU', 'DMUHU', 'DMUL', 'NEG'
            , 'NMADD', 'NMSUB', 'RECIP', 'RINT', 'ROTR', 'ROUND', 'RSQRT'
            , 'SLL', 'SLLV', 'SQRT', 'SRA', 'SRAV', 'SRL', 'SRLV'
            , 'SUB', 'SUBU', 'madd', 'maddu', 'msub', 'msubu', 'sll'
            , 'srl', 'sra', 'sllv', 'srla', 'srlv']

        mips_mnemonics = [s.lower() for s in mips_mnemonics]
        arm_mnemonics = [s.lower() for s in arm_mnemonics]
        x86_mnemonics = [s.lower() for s in x86_mnemonics]

        count = 0
        for i in list(self.instructions):
            if self.architecture == 'x86':
                if str(i['mnemonic']).lower() in x86_mnemonics:
                    count = count + 1
            elif self.architecture == 'mips':
                if str(i['mnemonic']).lower() in mips_mnemonics:
                    count = count + 1
            elif self.architecture == 'arm':
                if str(i['mnemonic']).lower() in arm_mnemonics:
                    count = count + 1
            elif self.architecture == 'arm':
                if str(i['mnemonic']).lower() in arm_mnemonics:
                    count = count + 1
        nop = 0
        return count

    # Questa funzione conta le istruzioni logiche all'interno del blocco
    def countLogic(self):
        x86_mnemonics = ['and', 'andn', 'andnpd', 'andpd', 'andps', 'andnps', 'test', 'xor', 'xorpd', 'pslld'
            , 'ANDNPD', 'ANDNPS', 'ANDPD', 'ANDPS', 'KANDB', 'KANDD', 'KANDNB', 'KANDND', 'KANDNQ', 'KANDNW', 'KANDQ',
                         'KANDW'
            , 'KNOTB', 'KNOTq', 'KNOTD', 'KNOTw', 'korq', 'korb', 'korw', 'kord', 'KTESTB', 'ktestd', 'ktestq', 'ktestw'
            , 'KXNORB', 'KXNORd', 'KXNORq', 'KXORB', 'KXORq', 'KXORd', 'KXORw', 'NOT', 'OR', 'ORPD', 'ORPS', 'PAND',
                         'PAND'
            , 'PCMPEQB', 'PCMPEQD', 'PCMPEQQ', 'PCMPGTB', 'PTEST', 'pxor', 'VPCMPB', 'VPCMPD', 'VPCMPQ',
                         'VPTESTMB', 'VPTESTMD', 'VPTESTMQ', 'VPTESTMW', 'VPTESTNMB', 'VPTESTNMD', 'VPTESTNMQ',
                         'VPTESTNMW'
            , 'XORPD', 'XORPS']
        arm_mnemonics = ['AND', 'EOR', 'ORR', 'ORN', 'BIC']
        mips_mnemonics = ['and', 'andi', 'or', 'ori', 'xor', 'nor', 'slt', 'slti', 'sltu']

        mips_mnemonics = [s.lower() for s in mips_mnemonics]
        arm_mnemonics = [s.lower() for s in arm_mnemonics]
        x86_mnemonics = [s.lower() for s in x86_mnemonics]

        count = 0
        for i in list(self.instructions):
            if self.architecture == 'x86':
                if str(i['mnemonic']).lower() in x86_mnemonics:
                    count = count + 1
            elif self.architecture == 'mips':
                if str(i['mnemonic']).lower() in mips_mnemonics:
                    count = count + 1
            elif self.architecture == 'arm':
                if str(i['mnemonic']).lower() in arm_mnemonics:
                    count = count + 1
        return count

    def countTransfer(self):
        x86_mnemonics = ['BNDLDX', 'BNDMK', 'BNDMOV', 'BNDSTX'
            , 'CMOVA', 'CMOVZ', 'CMOVPO', 'CMOVPE', 'CMOVP', 'CMOVO', 'CMOVNZ', 'CMOVNP', 'CMOVNO', 'CMOVNG', 'CMOVL'
            , 'FIST', 'FISTP', 'FISTTP', 'FSAVE', 'KMOVB', 'KMOVD', 'KMOVQ', 'KMOVW'
            , 'LDDQU', 'LDS', 'LEA', 'LODS', 'LODSB', 'LODSD', 'LODSQ', 'LODSW'
            , 'LSS', 'LSL', 'MOV', 'MOVAPD', 'MOVAPS', 'MOVBE', 'MOVD', 'MOVDDUP', 'MOVDQ2Q', 'MOVDQA', 'MOVDQU'
            , 'MOVHLPS', 'MOVHPD', 'MOVHPS', 'MOVLHPS', 'MOVLPD', 'MOVLPS', 'MOVQ', 'MOVS', 'MOVSB', 'MOVSD', 'MOVNTQ'
            , 'MOVNTDQ', 'MOVMSKPS', 'MOVSQ', 'MOVSS', 'MOVSW', 'MOVSX', 'MOVSXD', 'MOVUPD', 'MOVUPS', 'MOVZX',
                         'PMOVMSKB'
            , 'PMOVSX', 'PMOVZX', 'PUSH', 'PUSHA', 'PUSHAD', 'PUSHF', 'STOS', 'STOSB', 'STOSD', 'STOSQ', 'STOSW'
            , 'VBROADCAST', 'VEXPANDPD', 'VEXPANDPS', 'VMOVDQA32', 'VMOVDQA64', 'VMOVDQU16', 'VMOVDQU32', 'VMOVDQU64',
                         'VMOVDQU8'
            , 'VPBROADCAST', 'VPBROADCASTB', 'VPEXPANDD', 'VPEXPANDQ', 'movb', 'movq']
        arm_mnemonics = ['MOV', 'MVN', 'MOVT', 'MRA', 'MAR', 'LDR', 'STR', 'PLD', 'PLI', 'PLDW', 'LDM', 'LDREX',
                         'LDREXD', 'STM', 'STREX', 'STREXD']
        mips_mnemonics = ['LB', 'LBE', 'LBU', 'LBUE', 'LD', 'LDE', 'LDU', 'LDUE', 'LDC1', 'LDC2'
            , 'LDL', 'LDPC', 'LDR', 'LDXC1', 'LH', 'LHE', 'LHU', 'LHUE', 'LL'
            , 'LLD', 'LLE', 'LLDP', 'LLWP', 'LLWPE', 'LSA', 'LUXC1', 'LW'
            , 'LWC1', 'LWC2', 'LWL', 'LWLE', 'LWPC'
            , 'LWR', 'LWRE', 'LWU', 'MOV', 'SB', 'SBE', 'SC'
            , 'SCD', 'SCDP', 'SCE', 'SCWP', 'SCWPE'
            , 'SD', 'SDBBP', 'SDC1', 'SDC2', 'SDL', 'SDR', 'SDXC1', 'SH', 'SHU', 'SHE'
            , 'SW', 'SWE', 'SWC1', 'SWC2', 'SWL', 'SWR', 'SWLE', 'SWRE', 'SWXC1']

        mips_mnemonics = [s.lower() for s in mips_mnemonics]
        arm_mnemonics = [s.lower() for s in arm_mnemonics]
        x86_mnemonics = [s.lower() for s in x86_mnemonics]

        count = 0
        for i in list(self.instructions):
            if self.architecture == 'x86':
                if str(i['mnemonic']).lower() in x86_mnemonics:
                    count = count + 1
            elif self.architecture == 'mips':
                if str(i['mnemonic']).lower() in mips_mnemonics:
                    count = count + 1
            elif self.architecture == 'arm':
                if str(i['mnemonic']).lower() in arm_mnemonics:
                    count = count + 1
        return count

    def extractConstansStrings(self):
        constants = 0
        strings = 0
        for i, ins in enumerate(self.instructions):
            if 'opex' not in ins:
                continue
            for operand in ins['opex']['operands']:
                if operand['type'] == 'imm':
                    if 'disasm' in self.r2_disasm[i] and 'str.' in self.r2_disasm[i]['disasm']:
                        strings += 1
                    elif operand['value'] in self.string_addr:
                        strings += 1
                    else:
                        constants += 1

        return (constants, strings)
