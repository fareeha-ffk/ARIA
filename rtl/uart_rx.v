`timescale 1ns/1ps

module uart_rx (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,
    output reg  [7:0] data_out,
    output reg        data_valid
);

parameter CLKS_PER_BIT = 434;

localparam IDLE  = 2'd0;
localparam START = 2'd1;
localparam DATA  = 2'd2;
localparam STOP  = 2'd3;

reg [1:0]  state     = IDLE;
reg [8:0]  clk_count = 0;
reg [2:0]  bit_index = 0;
reg [7:0]  rx_byte   = 0;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state      <= IDLE;
        clk_count  <= 0;
        bit_index  <= 0;
        data_valid <= 0;
        data_out   <= 0;
        rx_byte    <= 0;
    end
    else begin
        data_valid <= 0;
        case (state)
            IDLE: begin
                clk_count <= 0;
                bit_index <= 0;
                if (rx == 1'b0)
                    state <= START;
            end
            START: begin
                if (clk_count == (CLKS_PER_BIT/2) - 1) begin
                    clk_count <= 0;
                    if (rx == 1'b0)
                        state <= DATA;
                    else
                        state <= IDLE;
                end
                else
                    clk_count <= clk_count + 1;
            end
            DATA: begin
                if (clk_count == CLKS_PER_BIT - 1) begin
                    clk_count          <= 0;
                    rx_byte[bit_index] <= rx;
                    if (bit_index == 3'd7) begin
                        bit_index <= 0;
                        state     <= STOP;
                    end
                    else
                        bit_index <= bit_index + 1;
                end
                else
                    clk_count <= clk_count + 1;
            end
            STOP: begin
                if (clk_count == CLKS_PER_BIT - 1) begin
                    data_valid <= 1;
                    data_out   <= rx_byte;
                    clk_count  <= 0;
                    state      <= IDLE;
                end
                else
                    clk_count <= clk_count + 1;
            end
            default: state <= IDLE;
        endcase
    end
end

endmodule
// Updated for Week 1 & 2
