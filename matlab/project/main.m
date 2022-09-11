function main()
    % create ui programatically
    screen_sz = get(0,'screensize');
    screen_w = screen_sz(3);
    screen_h = screen_sz(4);
    
    fig = uifigure('Name', 'Matlab semester project');
    fig.Position = [0 0 screen_w screen_h];
    fig.WindowState = 'maximized';
    
    ax_w = round(3 * screen_w / 4);
    ax_h = round(4 * screen_h / 5);
    ax = uiaxes(fig, 'Position', [20 40 ax_w ax_h]);
    hold(ax, 'on');
    axis(ax, 'equal');
    view(ax, [-1 1 -0.5]);
    axis(ax, [-500 3000 -2300 2300 -500 2000]);
   
    col_x = ax_w + 40;
    col_w = round(screen_w / 5);
    
    txt = uieditfield(fig, 'Position', [col_x (ax_h - 10) col_w 40]);
    txt.FontSize = 20;
    txt.Value = 'cad_data.stp';
    
    bg = uibuttongroup(fig, 'Position', [col_x (ax_h - 60) col_w 40]);   
    rb1 = uiradiobutton(bg, 'Position', [10 10 80 20]);
    rb2 = uiradiobutton(bg, 'Position', [110 10 80 20]);
    rb1.Text = '2 doors';
    rb2.Text = '4 doors';
    
    btn = uibutton(fig, 'Text', 'LOAD', 'Position', [col_x (ax_h - 110) col_w 40]);
    btn.ButtonPushedFcn = @(~,~) loadd();
    
    chbx1 = uicheckbox(fig, 'Position', [col_x + 10 (ax_h - 150) 80 20]);
    chbx2 = uicheckbox(fig, 'Position', [(col_x + round((col_w - 20) / 2)) (ax_h - 150) 80 20]);
    chbx3 = uicheckbox(fig, 'Position', [col_x + 10 (ax_h - 180) 80 20]);
    chbx4 = uicheckbox(fig, 'Position', [(col_x + round((col_w - 20) / 2)) (ax_h - 180) 80 20]);
    chbx1.Text = 'Door 1';
    chbx2.Text = 'Door 2';
    chbx3.Text = 'Door 3';
    chbx4.Text = 'Door 4';
    chbx1.Visible = 'off';
    chbx2.Visible = 'off';
    chbx3.Visible = 'off';
    chbx4.Visible = 'off';
    chbx1.ValueChangedFcn = @(~,~) calculate_doors();
    chbx2.ValueChangedFcn = @(~,~) calculate_doors();
    chbx3.ValueChangedFcn = @(~,~) calculate_doors();
    chbx4.ValueChangedFcn = @(~,~) calculate_doors();
    
    s = uislider(fig, 'Position', [(col_x + 5) (ax_h - 210) (col_w - 10) 3]);
    s.Limits = [0 90];
    s.MajorTicks = 0:10:90;
    s.ValueChangingFcn = @(~,ev) redraw(ev.Value);
    s.Visible = 'off';
    
    [doors, h_g, d_f_g, d_r_g, s_f_g, s_r_g, front_door, rear_door,...
        front_hinges, rear_hinges, front_sensors, rear_sensors] = deal([]);
     
    function loadd()
        path = txt.Value;
        if isfile(path)
            if rb1.Value
                doors_n = [1 1];
            else
                doors_n = [1 1 1 1];
            end
            [front_door_n, rear_door_n, front_hinges_n, rear_hinges_n, front_sensors_n, rear_sensors_n] = parse(path, doors_n);
            if isempty(front_door_n) || rb2.Value && isempty(rear_door_n)
                errordlg("Could not load the desired amount of doors!", "Error");                                
            else
                doors = doors_n;
                [front_door, rear_door, front_hinges, rear_hinges, front_sensors, rear_sensors] = deal(front_door_n, rear_door_n, front_hinges_n, rear_hinges_n, front_sensors_n, rear_sensors_n);
                save_rot();
                clear_ax();
                h_g = plot_hinge(ax, front_hinges, rear_hinges, doors);
                [d_f_g, d_r_g, s_f_g, s_r_g] = plot_door(ax, 0, front_door, rear_door, front_hinges, rear_hinges, front_sensors, rear_sensors, doors);
                s.Visible = 'on';
                s.Value = 0;
                chbx1.Visible = 'on';
                chbx1.Value = 1;
                chbx2.Visible = 'on';
                chbx2.Value = 1;
                if numel(doors) == 2
                    chbx3.Visible = 'off';
                    chbx4.Visible = 'off';
                else
                    chbx3.Visible = 'on';
                    chbx3.Value = 1;
                    chbx4.Visible = 'on';
                    chbx4.Value = 1;
                end
            end
        else
            errordlg("Not a valid file path!", "Error");
        end
    end

    function calculate_doors()
        if numel(doors) == 2
            doors = [chbx1.Value chbx2.Value];
        else
            doors = [chbx1.Value chbx2.Value chbx3.Value chbx4.Value];
        end
        redraw(s.Value);
    end

    function redraw(value)
        clear_ax();
        h_g = plot_hinge(ax, front_hinges, rear_hinges, doors);
        [d_f_g, d_r_g, s_f_g, s_r_g] = plot_door(ax, value, front_door, rear_door, front_hinges, rear_hinges, front_sensors, rear_sensors, doors);
    end

    function clear_ax()
        delete(h_g);
        delete(d_f_g);
        delete(d_r_g);
        delete(s_f_g);
        delete(s_r_g);
    end

    function save_rot()
        fid = fopen('rot.txt','wt+');
        fprintf(fid, 'number of doors\n');
        fprintf(fid, '2 points left front sensors\n');
        fprintf(fid, 'matrix W used in front door rot matrix\n');
        if numel(doors) == 4
            fprintf(fid, '2 points left rear sensors\n');
            fprintf(fid, 'matrix W used in rear door rot matrix\n');
        end
        fprintf(fid, 'rotation matrix function\n');
        fprintf(fid, '%d\n', numel(doors));
        [W_f, R] = rotate(front_hinges(2, :) - front_hinges(1, :));
        fprintf(fid, '%s\n', mat2str(front_sensors));
        fprintf(fid, '%s\n', mat2str(W_f));
        if numel(doors) == 4
            [W_r, ~] = rotate(rear_hinges(2, :) - rear_hinges(1, :));
            fprintf(fid, '%s\n', mat2str(rear_sensors));
            fprintf(fid, '%s\n', mat2str(W_r));
        end
        
        fprintf(fid, '%s\n', func2str(R));
        fclose(fid);
    end
end