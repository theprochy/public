function [d_f_g, d_r_g, s_f_g, s_r_g] = plot_door(ax, angle, d_f, d_r, h_f, h_r, s_f, s_r, doors)
    c = [0.902 0.902 0.902];    

    % front door(s)
    h_f_v = h_f(2, :) - h_f(1, :);
    [~, R_f] = rotate(h_f_v);
    offset_f = (h_f(1, :) + h_f(2, :)) / 2; 
    d_f_rot = (R_f(-angle*pi/180)*(d_f - offset_f).').' + offset_f;
    s_f_rot = (R_f(-angle*pi/180)*(s_f - offset_f).').' + offset_f;
    s_f_rot = [s_f_rot; s_f_rot(:, 1) -s_f_rot(:, 2) s_f_rot(:, 3)];
    s_f_rot = s_f_rot(logical(repelem(doors(1:2), 2)), :);
    y = [d_f_rot(:, 2) -d_f_rot(:, 2)];
    y = y(:,logical(doors(1:2)));
    if isempty(y)
        d_f_g = [];
    else
        d_f_g = fill3(ax, d_f_rot(:, 1), y, d_f_rot(:, 3), c);
        set(d_f_g,'facealpha',.8);
    end
    if isempty(s_f_rot)
        s_f_g = [];
    else
        s_f_g = scatter3(ax, s_f_rot(:, 1), s_f_rot(:, 2), s_f_rot(:, 3), 20, 'red', 's', 'filled');
    end
    
    % rear door(s)
    if numel(doors) == 4
        h_r_v = h_r(2, :) - h_r(1, :);
        [~, R_r] = rotate(h_r_v);
        offset_r = (h_r(1, :) + h_r(2, :)) / 2;
        d_r_rot = (R_r(-angle*pi/180)*(d_r - offset_r).').' + offset_r;
        s_r_rot = (R_f(-angle*pi/180)*(s_r - offset_r).').' + offset_r;
        s_r_rot = [s_r_rot; s_r_rot(:, 1) -s_r_rot(:, 2) s_r_rot(:, 3)];
        s_r_rot = s_r_rot(logical(repelem(doors(3:4), 2)), :);
        y = [d_r_rot(:, 2) -d_r_rot(:, 2)];
        y = y(:,logical(doors(3:4)));
        if isempty(y)
            d_r_g = [];
        else
            d_r_g = fill3(ax, d_r_rot(:, 1), y, d_r_rot(:, 3), c);
            set(d_r_g,'facealpha',.8);
        end
        if isempty(s_r_rot)
            s_r_g = [];
        else
            s_r_g = scatter3(ax, s_r_rot(:, 1), s_r_rot(:, 2), s_r_rot(:, 3), 20, 'red', 's', 'filled');
        end
    else
        d_r_g = [];
        s_r_g = [];
    end
end