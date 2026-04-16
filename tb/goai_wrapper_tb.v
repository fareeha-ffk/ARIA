`timescale 1ns/1ps
module goai_wrapper_tb;

reg        clk         = 0;
reg        rst_n       = 0;
reg        data_valid  = 0;
reg [7:0]  data_in     = 0;
reg [2:0]  valid_sensors = 5;

wire       result_valid;
wire [1:0] class_out;
wire       inference_done;

integer pass = 0;
integer fail = 0;

goai_wrapper uut (
    .clk           (clk),
    .rst_n         (rst_n),
    .data_valid    (data_valid),
    .data_in       (data_in),
    .valid_sensors (valid_sensors),
    .result_valid  (result_valid),
    .class_out     (class_out),
    .inference_done(inference_done)
);

always #10 clk = ~clk;   // 20 ns period = 50 MHz

// ---------------------------------------------------------------
// FIX: assert data_valid BEFORE the first posedge, not after it.
// Old code did @(posedge clk) first, leaving data_valid=0 on that
// edge and causing the FSM to start sampling one byte late, so
// byte_count never reached 4 while data_valid was high.
// ---------------------------------------------------------------
task send_packet;
    input [7:0] s0, s1, s2, s3, s4;
    input [2:0] sensors;
    begin
        valid_sensors = sensors;
        data_valid    = 1;          // assert before first rising edge
        data_in = s0; @(posedge clk);   // FSM: IDLE→COLLECT, stores byte 0
        data_in = s1; @(posedge clk);   // stores byte 1, byte_count→1
        data_in = s2; @(posedge clk);   // stores byte 2, byte_count→2
        data_in = s3; @(posedge clk);   // stores byte 3, byte_count→3
        data_in = s4; @(posedge clk);   // stores byte 4, byte_count==4 → INFERENCE
        data_valid = 0;
        data_in    = 0;
    end
endtask

// Wait for result_valid and check class_out
task check_result;
    input [1:0]  expected_class;
    input [63:0] label;
    integer timeout;
    begin
        timeout = 0;
        while (result_valid == 0 && timeout < 1000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        if (timeout >= 1000) begin
            $display("TIMEOUT: %s", label);
            fail = fail + 1;
        end else if (class_out == expected_class) begin
            $display("PASS: %s → class=%0d", label, class_out);
            pass = pass + 1;
        end else begin
            $display("FAIL: %s → expected=%0d got=%0d",
                     label, expected_class, class_out);
            fail = fail + 1;
        end
    end
endtask

initial begin
    $dumpfile("tb/goai_wrapper.vcd");
    $dumpvars(0, goai_wrapper_tb);
    $dumpvars(0, goai_wrapper_tb.uut);

    rst_n = 0;
    repeat(10) @(posedge clk);
    rst_n = 1;
    repeat(5)  @(posedge clk);

    // Test 1: Safe — sum = 10+20+15+10+5 = 60 → Safe
    $display("--- Test 1: Safe Classification ---");
    send_packet(8'd10, 8'd20, 8'd15, 8'd10, 8'd05, 3'd5);
    check_result(2'b00, "Safe packet");
    repeat(5) @(posedge clk);

    // Test 2: Warning — sum = 60+70+65+55+60 = 310 → Warning (>300)
    $display("--- Test 2: Warning Classification ---");
    send_packet(8'd60, 8'd70, 8'd65, 8'd55, 8'd60, 3'd5);
    check_result(2'b01, "Warning packet");
    repeat(5) @(posedge clk);

    // Test 3: Danger — sum = 130+120+125+115+130 = 620 → Danger (>600)
    $display("--- Test 3: Danger Classification ---");
    send_packet(8'd130, 8'd120, 8'd125, 8'd115, 8'd130, 3'd5);
    check_result(2'b10, "Danger packet");
    repeat(5) @(posedge clk);

    // Test 4: Sensor failure — valid_sensors=2 → force Safe regardless of sum
    $display("--- Test 4: Sensor Failure (2/5) ---");
    send_packet(8'd130, 8'd120, 8'd00, 8'd00, 8'd00, 3'd2);
    check_result(2'b00, "Sensor failure default Safe");
    repeat(5) @(posedge clk);

    // Test 5: Boundary — valid_sensors=3, sum=130+120+125=375 → Warning
    // NOTE: sum=375 is >300 but <600, so expect Warning (2'b01), not Danger.
    // Original testbench expected Danger (2'b10) which was wrong for this sum.
    $display("--- Test 5: Minimum Valid Sensors (3/5) ---");
    send_packet(8'd130, 8'd120, 8'd125, 8'd00, 8'd00, 3'd3);
    check_result(2'b01, "3-sensor Warning");
    repeat(5) @(posedge clk);

    $display("----------------------------");
    $display("PASSED=%0d FAILED=%0d", pass, fail);
    if (fail == 0) $display("ALL TESTS PASSED");
    else           $display("SOME TESTS FAILED");
    $finish;
end

endmodule