`timescale 1ns/1ps
module output_fsm_tb;

reg        clk          = 0;
reg        rst_n        = 0;
reg        result_valid = 0;
reg [1:0]  class_in     = 0;

wire       led_safe;
wire       led_warning;
wire       led_danger;
wire       alert_out;
wire [1:0] air_quality;

integer pass = 0, fail = 0;

output_fsm uut (
    .clk         (clk),
    .rst_n       (rst_n),
    .result_valid(result_valid),
    .class_in    (class_in),
    .led_safe    (led_safe),
    .led_warning (led_warning),
    .led_danger  (led_danger),
    .alert_out   (alert_out),
    .air_quality (air_quality)
);

always #10 clk = ~clk;

task tick; begin @(posedge clk); #1; end endtask

task send_result;
    input [1:0] cls;
    begin
        class_in     = cls;
        result_valid = 1;
        tick;
        result_valid = 0;
        tick; tick;
    end
endtask

task check_outputs;
    input exp_safe, exp_warn, exp_danger, exp_alert;
    input [1:0] exp_state;
    input [63:0] label;
    begin
        if (led_safe    === exp_safe   &&
            led_warning === exp_warn   &&
            led_danger  === exp_danger &&
            alert_out   === exp_alert  &&
            air_quality === exp_state) begin
            $display("PASS: %s", label);
            pass = pass + 1;
        end else begin
            $display("FAIL: %s  safe=%b warn=%b danger=%b alert=%b state=%0d",
                     label, led_safe, led_warning, led_danger,
                     alert_out, air_quality);
            fail = fail + 1;
        end
    end
endtask

initial begin
    $dumpfile("tb/output_fsm.vcd");
    $dumpvars(0, output_fsm_tb);

    rst_n = 0; repeat(5) tick;
    rst_n = 1; repeat(2) tick;

    $display("--- Test 1: Reset state ---");
    check_outputs(1,0,0,0, 2'd0, "Safe after reset");

    $display("--- Test 2: Warning ---");
    send_result(2'b01);
    check_outputs(0,1,0,0, 2'd1, "Warning state");

    $display("--- Test 3: Danger ---");
    send_result(2'b10);
    check_outputs(0,0,1,1, 2'd2, "Danger + alert");

    $display("--- Test 4: Back to Safe ---");
    send_result(2'b00);
    check_outputs(1,0,0,0, 2'd0, "Safe again");

    $display("--- Test 5: No update without result_valid ---");
    send_result(2'b01);
    class_in = 2'b10;
    result_valid = 0;
    tick; tick; tick;
    check_outputs(0,1,0,0, 2'd1, "State held without result_valid");

    $display("--- Test 6: alert_out asserts with Danger ---");
    send_result(2'b10);
    if (alert_out === 1 && led_danger === 1) begin
        $display("PASS: alert_out asserted in Danger");
        pass = pass + 1;
    end else begin
        $display("FAIL: alert_out not asserted in Danger");
        fail = fail + 1;
    end

    $display("----------------------------");
    $display("PASSED=%0d FAILED=%0d", pass, fail);
    if (fail == 0) $display("ALL TESTS PASSED");
    else           $display("SOME TESTS FAILED");
    $finish;
end

// SVA: Danger state must always have alert_out high (combinatorial, no lag)
always @(posedge clk) begin
    if (rst_n && air_quality === 2'd2 && alert_out !== 1)
        $display("SVA FAIL at %0t: Danger but alert_out=0", $time);
    if (rst_n && (led_safe + led_warning + led_danger) > 1)
        $display("SVA FAIL at %0t: multiple LEDs on", $time);
end

endmodule
