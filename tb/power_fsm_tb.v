`timescale 1ns/1ps
module power_fsm_tb;

reg  clk         = 0;
reg  rst_n       = 0;
reg  uart_active = 0;

wire clk_en_core;
wire clk_en_fifo;
wire clk_en_peripheral;
wire [1:0] power_state;

integer pass = 0, fail = 0;

power_fsm uut (
    .clk             (clk),
    .rst_n           (rst_n),
    .uart_active     (uart_active),
    .clk_en_core     (clk_en_core),
    .clk_en_fifo     (clk_en_fifo),
    .clk_en_peripheral(clk_en_peripheral),
    .power_state     (power_state)
);

always #10 clk = ~clk;

task tick; begin @(posedge clk); #1; end endtask

task pulse_uart; begin
    uart_active = 1; tick;
    uart_active = 0;
end endtask

task check_state;
    input [1:0]  exp_state;
    input        exp_core, exp_fifo, exp_periph;
    input [63:0] label;
    begin
        if (power_state === exp_state &&
            clk_en_core === exp_core &&
            clk_en_fifo === exp_fifo &&
            clk_en_peripheral === exp_periph) begin
            $display("PASS: %s  state=%0d", label, power_state);
            pass = pass + 1;
        end else begin
            $display("FAIL: %s  state=%0d(exp %0d) core=%b fifo=%b periph=%b",
                     label, power_state, exp_state,
                     clk_en_core, clk_en_fifo, clk_en_peripheral);
            fail = fail + 1;
        end
    end
endtask

initial begin
    $dumpfile("tb/power_fsm.vcd");
    $dumpvars(0, power_fsm_tb);

    rst_n = 0; repeat(5) tick;
    rst_n = 1; repeat(2) tick;

    // Test 1: After reset should be ACTIVE, all clocks on
    $display("--- Test 1: Reset state ---");
    check_state(2'd0, 1, 1, 1, "ACTIVE after reset");

    // Test 2: No activity for IDLE_TIMEOUT cycles → should go IDLE
    $display("--- Test 2: ACTIVE to IDLE transition ---");
    repeat(35) tick;    // > IDLE_TIMEOUT(30)
    check_state(2'd1, 0, 1, 1, "IDLE state");

    // Test 3: UART pulse wakes back to ACTIVE
    $display("--- Test 3: IDLE to ACTIVE on uart_active ---");
    pulse_uart;
    tick; tick;
    check_state(2'd0, 1, 1, 1, "ACTIVE after wakeup");

    // Test 4: Let it go IDLE then SLEEP
    $display("--- Test 4: ACTIVE to IDLE to SLEEP ---");
    repeat(35) tick;    // → IDLE
    repeat(65) tick;    // > SLEEP_TIMEOUT(60) → SLEEP
    check_state(2'd2, 0, 0, 0, "SLEEP state");

    // Test 5: UART pulse from SLEEP wakes all the way to ACTIVE
    $display("--- Test 5: SLEEP to ACTIVE on uart_active ---");
    pulse_uart;
    tick; tick;
    check_state(2'd0, 1, 1, 1, "ACTIVE from SLEEP");

    $display("----------------------------");
    $display("PASSED=%0d FAILED=%0d", pass, fail);
    if (fail == 0) $display("ALL TESTS PASSED");
    else           $display("SOME TESTS FAILED");
    $finish;
end

endmodule
