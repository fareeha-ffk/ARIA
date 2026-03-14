`timescale 1ns/1ps
module async_fifo_tb;
parameter WR_PERIOD = 8680;
parameter RD_PERIOD = 20;
reg wr_clk=0, rd_clk=0;
reg wr_rst_n=0, rd_rst_n=0;
reg wr_en=0, rd_en=0;
reg [7:0] wr_data=0;
wire [7:0] rd_data;
wire full, empty;
integer pass=0, fail=0;
async_fifo #(.DATA_WIDTH(8),.ADDR_WIDTH(4)) uut (
    .wr_clk(wr_clk),.wr_rst_n(wr_rst_n),.wr_en(wr_en),
    .wr_data(wr_data),.full(full),
    .rd_clk(rd_clk),.rd_rst_n(rd_rst_n),.rd_en(rd_en),
    .rd_data(rd_data),.empty(empty)
);
always #(WR_PERIOD/2) wr_clk = ~wr_clk;
always #(RD_PERIOD/2) rd_clk = ~rd_clk;
task write_byte;
    input [7:0] data;
    begin
        @(posedge wr_clk);
        wr_en=1; wr_data=data;
        @(posedge wr_clk);
        wr_en=0;
    end
endtask
task read_check;
    input [7:0] expected;
    integer timeout;
    reg [7:0] captured;
    begin
        timeout=0;
        // Wait until not empty
        while(empty && timeout<500000) begin
            @(posedge rd_clk);
            timeout=timeout+1;
        end
        if(timeout>=500000) begin
            $display("TIMEOUT: 0x%02X",expected);
            fail=fail+1;
        end
        else begin
            // Pulse rd_en for one cycle
            @(negedge rd_clk);
            rd_en=1;
            @(negedge rd_clk);
            rd_en=0;
            // Wait two cycles for registered output to settle
            @(posedge rd_clk);
            @(posedge rd_clk);
            captured = rd_data;
            if(captured==expected) begin
                $display("PASS: wrote=0x%02X read=0x%02X",expected,captured);
                pass=pass+1;
            end
            else begin
                $display("FAIL: wrote=0x%02X read=0x%02X",expected,captured);
                fail=fail+1;
            end
        end
    end
endtask
initial begin
    $dumpfile("tb/async_fifo.vcd");
    $dumpvars(0,async_fifo_tb);
    wr_rst_n=0; rd_rst_n=0;
    repeat(10) @(posedge rd_clk);
    wr_rst_n=1; rd_rst_n=1;
    repeat(10) @(posedge rd_clk);
    $display("--- Test 1: Single Bytes ---");
    write_byte(8'hAA); read_check(8'hAA);
    write_byte(8'h55); read_check(8'h55);
    write_byte(8'hFF); read_check(8'hFF);
    write_byte(8'h00); read_check(8'h00);
    $display("--- Test 2: 9-Byte Packet ---");
    write_byte(8'hA5); write_byte(8'h01); write_byte(8'h2C);
    write_byte(8'h00); write_byte(8'h64); write_byte(8'h26);
    write_byte(8'h48); write_byte(8'h61); write_byte(8'hB3);
    read_check(8'hA5); read_check(8'h01); read_check(8'h2C);
    read_check(8'h00); read_check(8'h64); read_check(8'h26);
    read_check(8'h48); read_check(8'h61); read_check(8'hB3);
    $display("--- Test 3: Empty Flag ---");
    repeat(200) @(posedge rd_clk);
    if(empty) begin $display("PASS: FIFO empty"); pass=pass+1; end
    else begin $display("FAIL: should be empty"); fail=fail+1; end
    $display("--- Test 4: Full Flag ---");
    repeat(16) begin write_byte($random & 8'hFF); end
    repeat(20) @(posedge wr_clk);
    if(full) begin $display("PASS: FIFO full"); pass=pass+1; end
    else begin $display("FAIL: should be full"); fail=fail+1; end
    $display("PASSED=%0d FAILED=%0d",pass,fail);
    if(fail==0) $display("ALL TESTS PASSED");
    $finish;
end
endmodule
