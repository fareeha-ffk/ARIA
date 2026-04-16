`timescale 1ns/1ps
module validity_reg_tb;

reg        clk          = 0;
reg        rst_n        = 0;
reg [5:0]  sensor_strobe = 0;

wire [5:0] valid_mask;
wire [2:0] active_count;

integer pass = 0, fail = 0;

validity_reg uut (
    .clk           (clk),
    .rst_n         (rst_n),
    .sensor_strobe (sensor_strobe),
    .valid_mask    (valid_mask),
    .active_count  (active_count)
);

always #10 clk = ~clk;

task tick; begin @(posedge clk); #1; end endtask

task check;
    input [5:0]  exp_mask;
    input [2:0]  exp_count;
    input [63:0] label;
    begin
        if (valid_mask === exp_mask && active_count === exp_count) begin
            $display("PASS: %s  mask=%06b count=%0d", label, valid_mask, active_count);
            pass = pass + 1;
        end else begin
            $display("FAIL: %s  mask=%06b(exp %06b) count=%0d(exp %0d)",
                     label, valid_mask, exp_mask, active_count, exp_count);
            fail = fail + 1;
        end
    end
endtask

initial begin
    $dumpfile("tb/validity_reg.vcd");
    $dumpvars(0, validity_reg_tb);

    rst_n = 0; repeat(5) tick;
    rst_n = 1; repeat(2) tick;

    // Test 1: all sensors strobing — all should stay valid
    $display("--- Test 1: All sensors active ---");
    sensor_strobe = 6'b111111;
    repeat(5) tick;
    sensor_strobe = 0;
    tick;
    check(6'b111111, 3'd6, "All active");

    // Test 2: kill sensor 0 and 1 by not strobing them past timeout
    $display("--- Test 2: Sensors 0,1 timeout ---");
    sensor_strobe = 6'b111100;   // only sensors 2-5 keep strobing
    repeat(25) tick;             // > TIMEOUT_CYCLES (20)
    sensor_strobe = 0;
    tick; tick;
    check(6'b111100, 3'd4, "Sensors 0,1 failed");

    // Test 3: revive sensor 0 with a strobe
    $display("--- Test 3: Revive sensor 0 ---");
    sensor_strobe = 6'b000001;
    tick;
    sensor_strobe = 0;
    tick; tick;
    check(6'b111101, 3'd5, "Sensor 0 revived");

    // Test 4: all sensors fail
    $display("--- Test 4: All sensors timeout ---");
    sensor_strobe = 0;
    repeat(25) tick;
    check(6'b000000, 3'd0, "All sensors failed");

    $display("----------------------------");
    $display("PASSED=%0d FAILED=%0d", pass, fail);
    if (fail == 0) $display("ALL TESTS PASSED");
    else           $display("SOME TESTS FAILED");
    $finish;
end

endmodule
