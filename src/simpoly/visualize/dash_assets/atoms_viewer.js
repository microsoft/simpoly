console.log('Loading...');

function get_viewer(id, config) {
    if (!(id in viewer_dict)) {
        // Create new viewer and store it in global dictionary
        let element = document.getElementById(id);
        if (!element) {
            throw 'Element with ID "' + id + '" not found';
        }
        viewer_dict[id] = $3Dmol.createViewer(element, config);
    }
    return viewer_dict[id];
}

function update_viewer(id, config) {
    let viewer = get_viewer(id, config.viewer);

    // Clear viewer
    viewer.removeAllLabels();
    viewer.removeAllModels();
    viewer.removeAllShapes();
    viewer.removeAllSurfaces();

    // Model
    let model = viewer.addModel(config.model.data, config.model.format);

    // Unit cell
    if (config.model.cell !== null && config.model.show_cell) {
        // Add crystal data manually since the XYZ format doesn't carry any
        const cell = config.model.cell;
        const mat = new $3Dmol.Matrix3(
            cell[0][0], cell[0][1], cell[0][2],
            cell[1][0], cell[1][1], cell[1][2],
            cell[2][0], cell[2][1], cell[2][2],
        );

        // Transpose matrix as 3DMol.js seems to be following a different convention
        mat.transpose();

        model.setCrystMatrix(mat);
        viewer.addUnitCell(model, { alabel: 'X', blabel: 'Y', clabel: 'Z' });
    }

    // Style
    for (let i = 0; i < config.styles.length; i++) {
        if (i === 0) {
            viewer.setStyle(config.styles[i].sel, config.styles[i].style)
        } else {
            viewer.addStyle(config.styles[i].sel, config.styles[i].style)
        }
    }

    // Labels
    for (let i = 0; i < config.labels.length; i++) {
        let label = config.labels[i];
        viewer.addLabel(label.text, label.options, label.sel, label.noshow);
    }

    // Arrows
    for (let i = 0; i < config.arrows.length; i++) {
        viewer.addArrow(config.arrows[i]);
    }

    // Lines
    for (let i = 0; i < config.lines.length; i++) {
        viewer.addLine(config.lines[i]);
    }

    // Zoom
    if (config.zoom) {
        viewer.zoomTo();
    }

    // Render
    viewer.render();

    return 0;
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        // Note: the function needs to return something (not null)
        update_viewer: update_viewer,
    }
});

let viewer_dict;

/* Since null == undefined is true, the following will catch both null and undefined */
if (viewer_dict == null) {
    console.log('Resetting viewers');
    viewer_dict = Object();
}

