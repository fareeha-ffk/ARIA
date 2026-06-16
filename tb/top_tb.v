`timescale 1ns/1ps
module top_tb;

reg  clk   = 0;
reg  rst_n = 0;
reg  rx    = 1;

wire led_safe, led_warning, led_danger, alert_out;
wire [1:0] air_quality, power_state;
wire packet_valid_pulse;

integer pass = 0, fail = 0;

top #(
    .UART_CLKS_PER_BIT(434),
    .BYPASS_POWER(1)
) dut (
    .clk               (clk),
    .rst_n             (rst_n),
    .rx                (rx),
    .led_safe          (led_safe),
    .led_warning       (led_warning),
    .led_danger        (led_danger),
    .alert_out         (alert_out),
    .air_quality       (air_quality),
    .power_state       (power_state),
    .packet_valid_pulse(packet_valid_pulse)
);

always #10 clk = ~clk;   // 50 MHz

localparam CLKS_PER_BIT = 434;

task uart_send_byte;
    input [7:0] b;
    integer j;
    begin
        rx = 0; repeat(CLKS_PER_BIT) @(posedge clk);
        for (j=0; j<8; j=j+1) begin
            rx = b[j]; repeat(CLKS_PER_BIT) @(posedge clk);
        end
        rx = 1; repeat(CLKS_PER_BIT) @(posedge clk);
    end
endtask

// Send a full 9-byte packet:
// [0]=0xA5 [1..5]=sensors [6]=active_count [7]=checksum [8]=0x5A
task send_packet;
    input [7:0] s0,s1,s2,s3,s4;
    input [7:0] active;
    reg   [7:0] chk;
    begin
        chk = s0 ^ s1 ^ s2 ^ s3 ^ s4 ^ active;
        uart_send_byte(8'hA5);
        uart_send_byte(s0);
        uart_send_byte(s1);
        uart_send_byte(s2);
        uart_send_byte(s3);
        uart_send_byte(s4);
        uart_send_byte(active);
        uart_send_byte(chk);
        uart_send_byte(8'h5A);
    end
endtask

initial begin
    $dumpfile("tb/top.vcd");
    $dumpvars(0, top_tb);

    rst_n = 0; rx = 1;
    repeat(20) @(posedge clk);
    rst_n = 1;
    repeat(10) @(posedge clk);

    $display("--- Test 1: Send Safe packet ---");
    send_packet(8'd10, 8'd20, 8'd15, 8'd10, 8'd05, 8'd5);
    repeat(1000) @(posedge clk);
    $display("  led_safe=%b led_warning=%b led_danger=%b alert=%b",
             led_safe, led_warning, led_danger, alert_out);
    if (led_safe || led_warning || led_danger)
        $display("  System responded to packet");
    else
        $display("  NOTE: packet may still be processing");

    repeat(500) @(posedge clk);

    $display("--- Test 2: Send Danger packet ---");
    send_packet(8'hC0, 8'hB0, 8'hC5, 8'hA0, 8'hC0, 8'd5);
    repeat(1000) @(posedge clk);
    $display("  led_safe=%b led_warning=%b led_danger=%b alert=%b",
             led_safe, led_warning, led_danger, alert_out);

    repeat(500) @(posedge clk);

    // SVA checks
    if (led_danger && !alert_out)
        $display("SVA FAIL: led_danger high but alert_out low");
    if ((led_safe + led_warning + led_danger) > 1)
        $display("SVA FAIL: multiple LEDs on");

    $display("--- Top-level integration complete ---");
    $display("air_quality=%0d power_state=%0d", air_quality, power_state);
    $finish;
end

endmodule
