IV  = iverilog
VVP = vvp
RTL = rtl
TB  = tb

.PHONY: all clean uart_rx async_fifo validity_reg power_fsm goai_wrapper output_fsm top golden status

all: uart_rx async_fifo validity_reg power_fsm goai_wrapper output_fsm top golden
	@echo ""
	@echo "ARIA pipeline complete"

uart_rx:
	@echo "[1/7] Testing uart_rx..."
	@$(IV) -o $(TB)/uart_rx_sim $(TB)/uart_rx_tb.v $(RTL)/uart_rx.v && $(VVP) $(TB)/uart_rx_sim 2>&1 | tee $(TB)/uart_rx.log
	@grep -q "PASS" $(TB)/uart_rx.log && echo "  uart_rx: PASS" || echo "  uart_rx: FAIL"

async_fifo:
	@echo "[2/7] Testing async_fifo..."
	@$(IV) -o $(TB)/async_fifo_sim $(TB)/async_fifo_tb.v $(RTL)/async_fifo.v && $(VVP) $(TB)/async_fifo_sim 2>&1 | tee $(TB)/async_fifo.log
	@grep -q "PASS" $(TB)/async_fifo.log && echo "  async_fifo: PASS" || echo "  async_fifo: FAIL"

validity_reg:
	@echo "[3/7] Testing validity_reg..."
	@$(IV) -o $(TB)/validity_reg_sim $(TB)/validity_reg_tb.v $(RTL)/validity_reg.v && $(VVP) $(TB)/validity_reg_sim 2>&1 | tee $(TB)/validity_reg.log
	@grep -q "PASS" $(TB)/validity_reg.log && echo "  validity_reg: PASS" || echo "  validity_reg: FAIL"

power_fsm:
	@echo "[4/7] Testing power_fsm..."
	@$(IV) -o $(TB)/power_fsm_sim $(TB)/power_fsm_tb.v $(RTL)/power_fsm.v && $(VVP) $(TB)/power_fsm_sim 2>&1 | tee $(TB)/power_fsm.log
	@grep -q "PASS" $(TB)/power_fsm.log && echo "  power_fsm: PASS" || echo "  power_fsm: FAIL"

goai_wrapper:
	@echo "[5/7] Testing goai_wrapper..."
	@$(IV) -o $(TB)/goai_wrapper_sim $(TB)/goai_wrapper_tb.v $(RTL)/goai_wrapper_simple.v && $(VVP) $(TB)/goai_wrapper_sim 2>&1 | tee $(TB)/goai_wrapper.log
	@grep -q "ALL TESTS PASSED" $(TB)/goai_wrapper.log && echo "  goai_wrapper: PASS" || echo "  goai_wrapper: FAIL"

output_fsm:
	@echo "[6/7] Testing output_fsm..."
	@$(IV) -o $(TB)/output_fsm_sim $(TB)/output_fsm_tb.v $(RTL)/output_fsm.v && $(VVP) $(TB)/output_fsm_sim 2>&1 | tee $(TB)/output_fsm.log
	@grep -q "ALL TESTS PASSED" $(TB)/output_fsm.log && echo "  output_fsm: PASS" || echo "  output_fsm: FAIL"

top:
	@echo "[7/7] Testing top-level..."
	@$(IV) -o $(TB)/top_sim $(TB)/top_tb.v $(RTL)/top.v $(RTL)/uart_rx.v $(RTL)/async_fifo.v $(RTL)/goai_wrapper_simple.v $(RTL)/validity_reg.v $(RTL)/power_fsm.v $(RTL)/output_fsm.v && $(VVP) $(TB)/top_sim 2>&1 | tee $(TB)/top.log
	@echo "  top: done"

golden:
	@echo "[+] Running golden vectors..."
	@$(IV) -o $(TB)/golden_sim $(TB)/top_golden_tb.v $(RTL)/top.v $(RTL)/uart_rx.v $(RTL)/async_fifo.v $(RTL)/goai_wrapper_simple.v $(RTL)/validity_reg.v $(RTL)/power_fsm.v $(RTL)/output_fsm.v && $(VVP) $(TB)/golden_sim 2>&1 | tee $(TB)/golden.log
	@grep -q "ZERO DISCREPANCY" $(TB)/golden.log && echo "  Golden: ZERO DISCREPANCY" || echo "  Golden: MISMATCHES"

status:
	@echo "--- ARIA Status ---"
	@for m in uart_rx async_fifo validity_reg power_fsm goai_wrapper output_fsm; do \
		if [ -f $(TB)/$$m.log ]; then \
			grep -q "PASS" $(TB)/$$m.log && echo "  PASS $$m" || echo "  FAIL $$m"; \
		else echo "  ---- $$m (not run)"; fi; done

clean:
	@rm -f $(TB)/*_sim $(TB)/*.log $(TB)/*.vcd
	@echo "Cleaned"
