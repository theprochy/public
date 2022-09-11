function hinges = plot_hinge(ax, h_f, h_r, doors)
    
    % plotting hinges
    [x, y, z] = deal([]);
    if doors(1)
        x = h_f(:,1);
        y = h_f(:, 2);
        z = h_f(:, 3);
    end
    if doors(2)
        x = [x; h_f(:,1)];  
        y = [y; -h_f(:, 2)];
        z = [z; h_f(:, 3)];
    end
    if numel(doors) == 4
        if doors(3)
            x = [x; h_r(:,1)];
            y = [y; h_r(:, 2)];
            z = [z; h_r(:, 3)];
        end
        if doors(4)
            x = [x; h_r(:,1)];
            y = [y; -h_r(:, 2)];
            z = [z; h_r(:, 3)];
        end
    end
    hinges = scatter3(ax, x, y, z, 15, 'blue', 'filled');
    
    % plotting rotational axes
    h_f_v = h_f(2, :) - h_f(1, :);
    l = [h_f(1,:)-1*h_f_v; h_f(1,:)+3*h_f_v];
    if doors(1)
        hinges = [hinges plot3(ax, l(:,1), l(:,2), l(:,3), '-.black')];
    end
    if doors(2)
        hinges = [hinges plot3(ax, l(:,1), -l(:,2), l(:,3), '-.black')];
    end
    if numel(doors) == 4
        h_r_v = h_r(2, :) - h_r(1, :);
        l = [h_r(1,:)-1*h_r_v; h_r(1,:)+3*h_r_v]; 
        if doors(3)
            hinges = [hinges plot3(ax, l(:,1), l(:,2), l(:,3), '-.black')];
        end
        if doors(4)
            hinges = [hinges plot3(ax, l(:,1), -l(:,2), l(:,3), '-.black')];
        end
    end
end