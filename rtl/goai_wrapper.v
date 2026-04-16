`timescale 1ns/1ps

module goai_wrapper (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        data_valid,
    input  wire [7:0]  data_in,
    input  wire [2:0]  valid_sensors,
    output reg         result_valid,
    output reg  [1:0]  class_out,
    output reg         inference_done
);

reg [7:0] sensors [0:4];
reg [2:0] byte_count;
reg [7:0] delay_count;
reg [1:0] state;

localparam IDLE = 0, COLLECT = 1, INFERENCE = 2, OUTPUT = 3;

initial begin
    state = IDLE;
    byte_count = 0;
    result_valid = 0;
    inference_done = 0;
    class_out = 0;
    delay_count = 0;
    sensors[0] = 0; sensors[1] = 0; sensors[2] = 0;
    sensors[3] = 0; sensors[4] = 0;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        byte_count <= 0;
        result_valid <= 0;
        inference_done <= 0;
        class_out <= 0;
        delay_count <= 0;
    end
    else begin
        result_valid <= 0;
        inference_done <= 0;
        
        case (state)
            IDLE: begin
                byte_count <= 0;
                if (data_valid) begin
                    sensors[0] <= data_in;
                    byte_count <= 1;
                    state <= COLLECT;
                end
            end
            
            COLLECT: begin
                if (data_valid) begin
                    sensors[byte_count] <= data_in;
                    if (byte_count == 4) begin
                        state <= INFERENCE;
                        delay_count <= 0;
                    end
                    byte_count <= byte_count + 1;
                end
            end
            
            INFERENCE: begin
                if (delay_count == 7) begin
                    state <= OUTPUT;
                end
                else begin
                    delay_count <= delay_count + 1;
                end
            end
            
            OUTPUT: begin
                // Dummy classifier
                if (valid_sensors < 3) begin
                    class_out <= 2'b00;  // Safe
                end
                else begin
                    integer sum;
                    sum = sensors[0] + sensors[1] + sensors[2] + sensors[3] + sensors[4];
                    if (sum > 600)
                        class_out <= 2'b10;  // Danger
                    else if (sum > 300)
                        class_out <= 2'b01;  // Warning
                    else
                        class_out <= 2'b00;  // Safe
                end
                result_valid <= 1;
                inference_done <= 1;
                state <= IDLE;
            end
        endcase
    end
end

endmodule