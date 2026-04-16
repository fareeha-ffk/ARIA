`timescale 1ns/1ps
// validity_reg.v
// Tracks which of the 6 sensor channels are currently active.
// A sensor is marked failed if it has not sent data within TIMEOUT_CYCLES.
// The valid_mask output feeds directly into goai_wrapper.v as valid_sensors.
module validity_reg (
    input  wire        clk,
    input  wire        rst_n,
    // One strobe per sensor channel when new data arrives
    input  wire [5:0]  sensor_strobe,
    // Output: bitmask of active sensors (1=active, 0=failed/timed-out)
    output reg  [5:0]  valid_mask,
    // Count of currently active sensors (feeds goai_wrapper valid_sensors)
    output reg  [2:0]  active_count
);

// How many clocks of silence before a sensor is declared failed.
// At 50 MHz and typical 100 ms sensor period: 5_000_000 cycles.
// Using 20 here for simulation speed — set to real value for synthesis.
localparam TIMEOUT_CYCLES = 20;

// Per-sensor watchdog counters
reg [7:0] timer [0:5];

integer i;

// Watchdog logic: reset timer on strobe, increment otherwise,
// mark failed when timer saturates.
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_mask   <= 6'b111111;   // assume all active at reset
        active_count <= 3'd6;
        for (i = 0; i < 6; i = i + 1)
            timer[i] <= 0;
    end else begin
        for (i = 0; i < 6; i = i + 1) begin
            if (sensor_strobe[i]) begin
                // Fresh data — sensor is alive, reset watchdog
                timer[i]      <= 0;
                valid_mask[i] <= 1'b1;
            end else if (timer[i] < TIMEOUT_CYCLES) begin
                timer[i] <= timer[i] + 1;
            end else begin
                // Timed out — mark sensor as failed
                valid_mask[i] <= 1'b0;
            end
        end

        // Recount active sensors every cycle
        active_count <= valid_mask[0] + valid_mask[1] + valid_mask[2]
                      + valid_mask[3] + valid_mask[4] + valid_mask[5];
    end
end

endmodule