`timescale 1ns/1ps
// power_fsm.v
// 3-state clock gating FSM.
// ACTIVE  : all clocks enabled, full processing.
// IDLE    : no recent UART data, peripheral clocks gated off.
// SLEEP   : extended inactivity, all clocks gated except wakeup logic.
// Transitions driven by activity timeout counters and uart_active strobe.
module power_fsm (
    input  wire clk,
    input  wire rst_n,
    // Pulse high for one cycle whenever uart_rx receives valid data
    input  wire uart_active,
    // Clock enable outputs — connect to clock gate cells in top.v
    output reg  clk_en_core,       // enables goai_wrapper + output_fsm
    output reg  clk_en_fifo,       // enables async_fifo write side
    output reg  clk_en_peripheral, // enables uart_rx, validity_reg
    // State output for debugging / coverage
    output reg [1:0] power_state
);

localparam ACTIVE = 2'd0;
localparam IDLE   = 2'd1;
localparam SLEEP  = 2'd2;

// Timeout thresholds (in clock cycles).
// Use small values here for simulation; scale up for real deployment.
localparam IDLE_TIMEOUT  = 30;   // cycles of silence before ACTIVE→IDLE
localparam SLEEP_TIMEOUT = 60;   // further cycles before IDLE→SLEEP

reg [7:0] idle_counter;
reg [7:0] sleep_counter;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        power_state      <= ACTIVE;
        clk_en_core      <= 1;
        clk_en_fifo      <= 1;
        clk_en_peripheral<= 1;
        idle_counter     <= 0;
        sleep_counter    <= 0;
    end else begin
        case (power_state)

            ACTIVE: begin
                clk_en_core       <= 1;
                clk_en_fifo       <= 1;
                clk_en_peripheral <= 1;
                if (uart_active) begin
                    idle_counter <= 0;           // reset on activity
                end else if (idle_counter < IDLE_TIMEOUT) begin
                    idle_counter <= idle_counter + 1;
                end else begin
                    power_state  <= IDLE;        // no data → go idle
                    idle_counter <= 0;
                end
            end

            IDLE: begin
                clk_en_core       <= 0;  // stop inference core
                clk_en_fifo       <= 1;  // keep FIFO alive (data might arrive)
                clk_en_peripheral <= 1;  // keep UART listening
                if (uart_active) begin
                    power_state   <= ACTIVE;     // wake up immediately
                    sleep_counter <= 0;
                end else if (sleep_counter < SLEEP_TIMEOUT) begin
                    sleep_counter <= sleep_counter + 1;
                end else begin
                    power_state   <= SLEEP;      // deep idle → sleep
                    sleep_counter <= 0;
                end
            end

            SLEEP: begin
                clk_en_core       <= 0;
                clk_en_fifo       <= 0;
                clk_en_peripheral <= 0;  // only wakeup detection runs
                if (uart_active) begin
                    power_state   <= ACTIVE;     // interrupt wakes system
                    idle_counter  <= 0;
                    sleep_counter <= 0;
                end
            end

            default: power_state <= ACTIVE;

        endcase
    end
end

endmodule
