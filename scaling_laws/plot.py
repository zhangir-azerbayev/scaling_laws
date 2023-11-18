from typing import List, Dict, Optional

AxisType = Literal['log', 'log2', 'log10' 'linear']

def scaling_scatter_3d(
    runs: Dict[str, List[float]],
    x_key: str, 
    y_key: str, 
    z_key: str = None,
    z_type: AxisType = 'log10',
    color_key: Optional[str] = None,
    color_type: AxisType = 'linear',
    fit_fn: Optional[Callable[[float, float], float]] = None,
    savepath: str = None,
):
    """
    Creates a 3D scaling plot in interactive HTML.

    Parameters:
    runs List[Dict[str, int]]: Each list entry is a point. Each point is a dictionary with named coordinates.
    x_key str: Coordinate name to plot on x axis. Default log scale.
    y_key str: Coordinate name to plot on y axis. Default log scale.
    z_key str: Coordinate name to plot on z axis.
    z_type AxisType
    color_key Optional[str]: Coordinate name to color points. If `None`, all points the same color
    color_type: AxisType
    fit_fn Optional[Callable[[float, float], float]]: Surface.
    surface str

    Returns:
    None

    Effects:
    HTML figure saved to `savepath`
    """
    x = runs[x_key]
    y = runs[y_key]
    z = runs[z_key]
    scatter = go.Scatter3d
    scatter_kwargs = dict(x=x, y=y, z=z)
    axis_kwargs = dict(
        scene=dict(
            xaxis=dict(type='log10', title=x_key),
            yaxis=dict(type='log10', title=y_key),
            zaxis=dict(type=z_type, title=z_key),
        )
    )
    hovertemplate=f"<b>{x_key}:%{{x:.2e}}</b><br><b>{y_key}:%{{y:.2e}}</b><br><b>{z_key}:%{{z:.2e}}</b>"
    
    if fit_fn:
        x_grid = np.logspace(np.log10(min(x)), np.log10(max(x)), 50)
        y_grid = np.logspace(np.log10(min(y)), np.log10(max(y)), 50)
        
        x_grid, y_grid = np.meshgrid(x_grid, y_grid)
        z_surface = fit_fn(x_grid, y_grid)
        
        
        surface = go.Surface(
            z=z_surface, 
            x=x_grid, 
            y=y_grid, 
            name='Surface', 
            opacity=0.6,
            contours={"z": {"show": True}},
            colorscale='Viridis'
        )
    else:
        surface = None

    if color_key:
        if color_type=="log":
            color_variable = np.log(runs[color_key])
        elif color_type=="log2":
            color_variable = np.log2(runs[color_key])
        elif color_type=="log10":
            color_variable = np.log10(runs[color_key])
        else:
            color_variable = runs[color_key]
            
        hovertemplate += f"<br><b>{color_key}:%{{marker.color:.2e}}</b><extra></extra>"
    else:
        color_variable = None
        hovertemplate += "<extra></extra>"
    
    data = [
        scatter(
            **scatter_kwargs,
            mode='markers',
            marker = dict(
                size=8,
                color=color_variable,
                colorscale='Viridis',
                colorbar=dict(title=color_key),
                opacity=0.8,
            ),
            hovertemplate=hovertemplate
        ), 
    ]
    
    if fit_fn:
        data.append(surface)
    
    fig = go.Figure(
        data=data
    )
    
    
    fig.update_layout(**axis_kwargs)
    
    fig.write_html(savepath)    


def scaling_scatter_2d(
    runs: Dict[str, List[float]],
    x_key: str, 
    x_type: AxisType = 'log10',
    y_key: str, 
    color_key: Optional[str] = None,
    color_type: AxisType = 'linear',
    fit_fn: Optional[Callable[[float], float]] = None,
    savepath: str = None,
):
    """
    Creates a 2D scaling plot and saves to HTML. If `color_key` is set, will draw isochromatic lines.

    Parameters:
    runs List[Dict[str, int]]: Each list entry is a point. Each point is a dictionary with named coordinates.
    x_key str: Coordinate name to plot on x axis. Default log10 scale.
    y_key str: Coordinate name to plot on y axis. Default log10 scale.
    color_key Optional[str]: Points of the same color will be connected by a line.
    color_type: AxisType
    fit_fn Optional[Callable[[float], float]]: Surface.
    surface str

    Returns:
    None

    Effects:
    HTML figure saved to `savepath`
    """
    id_of_float = lambda x: f'{x:.2e}'
    
    x = runs[x_key]
    y = runs[y_key]
    
    scatter = go.Scatter
    scatter_kwargs = dict(x=x, y=y)
    axis_kwargs = dict(
        xaxis=dict(type=x_type, title=x_key),
        yaxis=dict(type='log10', title=y_key),
    )
    hovertemplate=f"<b>{x_key}:%{{x:.2e}}</b><br><b>{y_key}:%{{y:.2e}}</b>"
    
    if color_key:
        if color_type=="log":
            color_variable = np.log(runs[color_key])
        elif color_type=="log2":
            color_variable = np.log2(runs[color_key])
        elif color_type=="log10":
            color_variable = np.log10(runs[color_key])
        else:
            color_variable = runs[color_key]
        
        cmax = max(color_variable)
        cmin = min(color_variable)
        viridis = mpl.colormaps['viridis']
        
        color_map = lambda x: plotly.colors.label_rgb(viridis((x-cmin)/(cmax-cmin))[:3])

            
        hovertemplate += f"<br><b>{color_key}:NAME</b><extra></extra>"
    else:
        color_variable = None
        cmin=None
        cmax=None
        hovertemplate += "<extra></extra>"
    
    fig = go.Figure()

    unique_color_ids = sorted(list(set(map(id_of_float, color_variable))), key=lambda x: float(x))
    
    for color_id in unique_color_ids:
        indices = [i for i,c in enumerate(color_variable) if id_of_float(c)==color_id]
        x_color = x[indices]
        y_color = y[indices]
        color = color_variable[indices[0]]
        color_name = color_map(color)
        
        fig.add_trace(go.Scatter(
            x=x_color,
            y=y_color,
            mode='lines+markers',
            name=color_id,
            marker = dict(
                size=8,
                color=color_name,
                opacity=0.8,
            ),
            line = dict(color=color_name),
            hovertemplate=hovertemplate.replace('NAME', f'{color:.2e}'),
        ))
    
    
    fig.update_layout(**axis_kwargs)
    
    fig.write_html(savepath)    
