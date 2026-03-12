`timescale 1ns/1ps
module uart_rx_tb;

parameter BIT_PERIOD = 8680;

reg        clk   = 0;
reg        rst_n = 0;
reg        rx    = 1;
wire [7:0] data_out;
wire       data_valid;
integer    pass  = 0;
integer    fail  = 0;

reg       cv = 0;
reg [7:0] cd = 0;

uart_rx uut (
    .clk(clk), .rst_n(rst_n), .rx(rx),
    .data_out(data_out), .data_valid(data_valid)
);

always #10 clk = ~clk;

always @(posedge clk)
    if (data_valid) begin
        cv <= 1;
        cd <= data_out;
    end

task send_and_check;
    input [7:0] b;
    integer i;
    begin
        cv = 0; cd = 0;
        rx = 0; #(BIT_PERIOD);
        for (i=0; i<8; i=i+1) begin
            rx = b[i]; #(BIT_PERIOD);
        end
        rx = 1; #(BIT_PERIOD*3);
        if      (cv==0)  begin $display("TIMEOUT 0x%02X",b); fail=fail+1; end
        else if (cd==b)  begin $display("PASS 0x%02X",b);    pass=pass+1; end
        else             begin $display("FAIL sent=0x%02X got=0x%02X",b,cd); fail=fail+1; end
    end
endtask

initial begin
    $dumpfile("tb/uart_rx.vcd");
    $dumpvars(0,uart_rx_tb);
    rst_n=0; rx=1;
    repeat(50) @(posedge clk);
    rst_n=1;
    repeat(50) @(posedge clk);

    // Single byte tests
    $display("--- Single Byte Tests ---");
    send_and_check(8'hAA);   // 10101010
    send_and_check(8'h55);   // 01010101
    send_and_check(8'hFF);   // 11111111
    send_and_check(8'h00);   // 00000000

    // 9-byte packet test
    $display("--- 9-Byte Packet Test ---");
    send_and_check(8'hA5);   // header
    send_and_check(8'h01);   // PM25 high
    send_and_check(8'h2C);   // PM25 low
    send_and_check(8'h00);   // VOC high
    send_and_check(8'h64);   // VOC low
    send_and_check(8'h26);   // HeatIdx
    send_and_check(8'h48);   // HR
    send_and_check(8'h61);   // SpO2
    send_and_check(8'hB3);   // CRC

    $display("----------------------------");
    $display("PASSED=%0d FAILED=%0d",pass,fail);
    if(fail==0) $display("ALL 13 TESTS PASSED");
    $finish;
end

endmodule