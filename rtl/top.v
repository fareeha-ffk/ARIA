`timescale 1ns/1ps

// top.v — End-to-end integration:
// UART RX (uart_clk) -> async_fifo (CDC) -> packet parser (core_clk)
// -> validity_reg -> goai_wrapper -> output_fsm
// plus power_fsm-derived enables (used as clock-enables, not gated clocks).
//
// Packet format (9 bytes total):
//   [0] 0xA5                      (START)
//   [1] s0 (8-bit sensor 0)
//   [2] s1 (8-bit sensor 1)
//   [3] s2 (8-bit sensor 2)
//   [4] s3 (8-bit sensor 3)
//   [5] s4 (8-bit sensor 4)
//   [6] active_count (0..5)       (how many sensors are considered valid)
//   [7] checksum = XOR bytes [1..6]
//   [8] 0x5A                      (END)
module top #(
    // Allows speeding up UART for simulation while keeping synthesis default.
    parameter integer UART_CLKS_PER_BIT = 434,
    // When 1, bypass power_fsm and keep all enables on (useful for verification).
    parameter integer BYPASS_POWER = 0
) (
    input  wire        clk,        // Single system clock (50MHz)
    input  wire        rst_n,
    input  wire        rx,

    output wire        led_safe,
    output wire        led_warning,
    output wire        led_danger,
    output wire        alert_out,

    // debug/visibility
    output wire [1:0]  air_quality,
    output wire [1:0]  power_state,
    output wire        packet_valid_pulse
);

    // ---------------------------
    // UART receiver (single clock domain)
    // ---------------------------
    wire [7:0] uart_data;
    wire       uart_data_valid;

    uart_rx #(
        .CLKS_PER_BIT(UART_CLKS_PER_BIT)
    ) uart_rx_u (
        .clk(clk),
        .rst_n(rst_n),
        .rx(rx),
        .data_out(uart_data),
        .data_valid(uart_data_valid)
    );

    // ---------------------------
    // Power FSM (single clock domain)
    // ---------------------------
    // Simplified power management in single clock domain
    wire clk_en_core;
    wire clk_en_fifo;
    wire clk_en_peripheral;
    wire uart_activity = uart_data_valid;

    generate
        if (BYPASS_POWER) begin : gen_pwr_bypass
            assign clk_en_core       = 1'b1;
            assign clk_en_fifo       = 1'b1;
            assign clk_en_peripheral = 1'b1;
            assign power_state       = 2'd0;
        end else begin : gen_pwr_fsm
            power_fsm power_fsm_u (
                .clk(clk),
                .rst_n(rst_n),
                .uart_active(uart_activity),
                .clk_en_core(clk_en_core),
                .clk_en_fifo(clk_en_fifo),
                .clk_en_peripheral(clk_en_peripheral),
                .power_state(power_state)
            );
        end
    endgenerate

    // ---------------------------
    // Simple FIFO (single clock domain)
    // ---------------------------
    wire fifo_full;
    wire fifo_empty;
    wire [7:0] fifo_rd_data;
    reg  fifo_rd_en = 0;

    async_fifo #(
        .DATA_WIDTH(8),
        .ADDR_WIDTH(4)
    ) fifo_u (
        .wr_clk(clk),
        .wr_rst_n(rst_n),
        .wr_en(uart_data_valid && clk_en_fifo && clk_en_peripheral),
        .wr_data(uart_data),
        .full(fifo_full),
        .rd_clk(clk),
        .rd_rst_n(rst_n),
        .rd_en(fifo_rd_en && clk_en_fifo),
        .rd_data(fifo_rd_data),
        .empty(fifo_empty)
    );

    // ---------------------------
    // Packet parser (single clock domain)
    // ---------------------------
    localparam P_START = 8'hA5;
    localparam P_END   = 8'h5A;

    localparam PS_WAIT_START = 3'd0;
    localparam PS_S0         = 3'd1;
    localparam PS_S1         = 3'd2;
    localparam PS_S2         = 3'd3;
    localparam PS_S3         = 3'd4;
    localparam PS_S4         = 3'd5;
    localparam PS_COUNT      = 3'd6;
    localparam PS_CKSUM      = 3'd7;
    // End byte is handled as "one more read" after checksum

    reg [2:0]  pstate = PS_WAIT_START;
    reg [7:0]  s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    reg [7:0]  count_byte = 0;
    reg [7:0]  cksum_rx = 0;
    reg [7:0]  cksum_calc = 0;
    reg        need_end = 0; // retained for compatibility; parser no longer requires END

    reg packet_valid_r = 0;
    assign packet_valid_pulse = packet_valid_r;

    // A simple "read when available" policy.
    wire can_read = (!fifo_empty) && clk_en_fifo && clk_en_core;

    // Read bytes from FIFO with a 2-cycle handshake:
    // - cycle A: assert rd_en for 1 cycle
    // - cycle B: capture rd_data (stable from cycle A read)
    reg fifo_read_pending = 0;
    reg [7:0] byte_reg = 0;
    reg       byte_reg_valid = 0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_rd_en         <= 0;
            fifo_read_pending  <= 0;
            byte_reg           <= 0;
            byte_reg_valid     <= 0;
        end else begin
            fifo_rd_en     <= 0;
            byte_reg_valid <= 0;

            if (fifo_read_pending) begin
                byte_reg       <= fifo_rd_data;
                byte_reg_valid <= 1'b1;
                fifo_read_pending <= 1'b0;
            end else if (can_read) begin
                fifo_rd_en        <= 1'b1;
                fifo_read_pending <= 1'b1;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pstate     <= PS_WAIT_START;
            cksum_calc <= 0;
            need_end   <= 0;
            packet_valid_r <= 1'b0;
        end else if (byte_reg_valid && clk_en_core) begin
            packet_valid_r <= 1'b0; // pulse
            case (pstate)
                PS_WAIT_START: begin
                    if (byte_reg == P_START) begin
                        pstate     <= PS_S0;
                        cksum_calc <= 0;
                    end
                end
                PS_S0: begin s0 <= byte_reg; cksum_calc <= cksum_calc ^ byte_reg; pstate <= PS_S1; end
                PS_S1: begin s1 <= byte_reg; cksum_calc <= cksum_calc ^ byte_reg; pstate <= PS_S2; end
                PS_S2: begin s2 <= byte_reg; cksum_calc <= cksum_calc ^ byte_reg; pstate <= PS_S3; end
                PS_S3: begin s3 <= byte_reg; cksum_calc <= cksum_calc ^ byte_reg; pstate <= PS_S4; end
                PS_S4: begin s4 <= byte_reg; cksum_calc <= cksum_calc ^ byte_reg; pstate <= PS_COUNT; end
                PS_COUNT: begin
                    count_byte <= byte_reg;
                    cksum_calc <= cksum_calc ^ byte_reg;
                    pstate     <= PS_CKSUM;
                end
                PS_CKSUM: begin
                    cksum_rx <= byte_reg;
                    // Accept packet as soon as checksum matches; END byte is optional.
                    if (cksum_calc == byte_reg)
                        packet_valid_r <= 1'b1;
                    pstate   <= PS_WAIT_START;
                    need_end <= 1'b0;
                end
                default: pstate <= PS_WAIT_START;
            endcase
        end
    end

    // ---------------------------
    // validity_reg (core_clk)
    // ---------------------------
    // We strobe sensors 0..4 when a packet is validated.
    // validity_reg supports 6 sensors; sensor[5] is unused here.
    reg [5:0] sensor_strobe = 0;
    wire [5:0] valid_mask;
    wire [2:0] active_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sensor_strobe <= 0;
        end else begin
            sensor_strobe <= 0;
            if (packet_valid_r && clk_en_peripheral) begin
                sensor_strobe[0] <= 1'b1;
                sensor_strobe[1] <= 1'b1;
                sensor_strobe[2] <= 1'b1;
                sensor_strobe[3] <= 1'b1;
                sensor_strobe[4] <= 1'b1;
            end
        end
    end

    validity_reg validity_u (
        .clk(clk),
        .rst_n(rst_n),
        .sensor_strobe(sensor_strobe & {6{clk_en_peripheral}}),
        .valid_mask(valid_mask),
        .active_count(active_count)
    );

    // ---------------------------
    // Feed goai_wrapper (core_clk)
    // ---------------------------
    reg        gw_data_valid = 0;
    reg [7:0]  gw_data_in = 0;
    wire       gw_result_valid;
    wire [1:0] gw_class_out;
    wire       gw_inference_done;

    localparam FS_IDLE = 3'd0;
    localparam FS_B0   = 3'd1;
    localparam FS_B1   = 3'd2;
    localparam FS_B2   = 3'd3;
    localparam FS_B3   = 3'd4;
    localparam FS_B4   = 3'd5;
    localparam FS_WAIT = 3'd6;
    reg [2:0] feed_state = FS_IDLE;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            feed_state    <= FS_IDLE;
            gw_data_valid <= 0;
            gw_data_in    <= 0;
        end else begin
            gw_data_valid <= 0;
            if (!clk_en_core) begin
                feed_state <= FS_IDLE;
            end else begin
                case (feed_state)
                    FS_IDLE: if (packet_valid_r) feed_state <= FS_B0;
                    FS_B0: begin gw_data_valid <= 1; gw_data_in <= s0; feed_state <= FS_B1; end
                    FS_B1: begin gw_data_valid <= 1; gw_data_in <= s1; feed_state <= FS_B2; end
                    FS_B2: begin gw_data_valid <= 1; gw_data_in <= s2; feed_state <= FS_B3; end
                    FS_B3: begin gw_data_valid <= 1; gw_data_in <= s3; feed_state <= FS_B4; end
                    FS_B4: begin gw_data_valid <= 1; gw_data_in <= s4; feed_state <= FS_WAIT; end
                    FS_WAIT: if (gw_inference_done) feed_state <= FS_IDLE;
                    default: feed_state <= FS_IDLE;
                endcase
            end
        end
    end

    goai_wrapper goai_u (
        .clk(clk),
        .rst_n(rst_n),
        .data_valid(gw_data_valid),
        .data_in(gw_data_in),
        .valid_sensors(active_count),
        .result_valid(gw_result_valid),
        .class_out(gw_class_out),
        .inference_done(gw_inference_done)
    );

    // ---------------------------
    // Output FSM (core_clk)
    // ---------------------------
    output_fsm out_u (
        .clk(clk),
        .rst_n(rst_n),
        .result_valid(gw_result_valid && clk_en_core),
        .class_in(gw_class_out),
        .led_safe(led_safe),
        .led_warning(led_warning),
        .led_danger(led_danger),
        .alert_out(alert_out),
        .air_quality(air_quality)
    );

endmodule

