`timescale 1ns/1ps
// output_fsm.v — outputs are combinatorial from state so SVA holds
// every cycle without a pipeline gap.
module output_fsm (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        result_valid,
    input  wire [1:0]  class_in,
    output reg         led_safe,
    output reg         led_warning,
    output reg         led_danger,
    output reg         alert_out,
    output reg [1:0]   air_quality
);

localparam SAFE    = 2'd0;
localparam WARNING = 2'd1;
localparam DANGER  = 2'd2;

// State register
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        air_quality <= SAFE;
    end else if (result_valid) begin
        case (class_in)
            2'b00:   air_quality <= SAFE;
            2'b01:   air_quality <= WARNING;
            2'b10:   air_quality <= DANGER;
            default: air_quality <= SAFE;
        endcase
    end
end

// Combinatorial output decode — no extra clock cycle lag
always @(*) begin
    led_safe    = 0;
    led_warning = 0;
    led_danger  = 0;
    alert_out   = 0;
    case (air_quality)
        SAFE:    led_safe    = 1;
        WARNING: led_warning = 1;
        DANGER: begin
                 led_danger  = 1;
                 alert_out   = 1;
                 end
        default: led_safe    = 1;
    endcase
end

endmodule
